"""CLI for evaluating the released question and permutation manifests."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import urllib.error

from .adapters import Adapter, payload_for, validate_config
from .protocol import ALL_ARMS, ORDER_ARMS, REPEAT_ARMS, InputError, input_sha256, keys_for, validate_order, validate_question, wire
from .scoring import score


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_inputs(question_path: Path | str, case_path: Path | str,
                permutation_path: Path | str, datasets: list[str] | None = None,
                limit: int | None = None) -> tuple[list[dict], dict]:
    """Load the selected cases and preserve their manifest order."""
    rows = read_jsonl(question_path)
    by_id = {}
    for row in rows:
        validate_question(row)
        if row["id"] in by_id:
            raise InputError("duplicate question ID")
        by_id[row["id"]] = row
    cases = read_jsonl(case_path)
    if datasets:
        cases = [row for row in cases if row.get("dataset") in datasets]
    if limit is not None:
        if limit < 1:
            raise InputError("--limit must be a positive integer")
        cases = cases[:limit]
    ids = [row["id"] for row in cases]
    if not ids or len(ids) != len(set(ids)):
        raise InputError("case manifest is empty or contains duplicate IDs")
    if not set(ids) <= set(by_id):
        raise InputError("question file does not cover the case manifest")
    questions = [by_id[qid] for qid in ids]
    for case, question in zip(cases, questions):
        if "input_sha256" in case and input_sha256(question) != case["input_sha256"]:
            raise InputError("Question content differs from the manifest; reconstruct the input file again")
        for field in ("dataset", "source_id", "source_config", "split", "option_count", "gold_index",
                      "source_labels", "subject", "difficulty", "initial_source_order"):
            if field in case and question.get(field) != case[field]:
                raise InputError("Question metadata differs from the manifest; reconstruct the input file again")
    selected_ids = set(ids)
    orders = {}
    for row in read_jsonl(permutation_path):
        qid = row["id"]
        if qid not in selected_ids:
            continue
        if qid in orders:
            raise InputError("duplicate permutation question ID")
        if qid not in by_id:
            raise InputError("permutation refers to an unknown question")
        supplied = row["orders"]
        if not isinstance(supplied, dict) or not set(ORDER_ARMS) <= set(supplied):
            raise InputError("permutation manifest must contain all eight reordered arms")
        n = len(by_id[qid]["options"])
        original = list(range(n))
        if supplied["original"] != original or supplied["reversed"] != original[::-1]:
            raise InputError("original/reversed orders are incorrect")
        supplied = dict(supplied)
        for arm in REPEAT_ARMS:
            if arm in supplied and supplied[arm] != original:
                raise InputError("original-order repeat is reordered")
            supplied[arm] = original.copy()
        if set(supplied) != set(ALL_ARMS):
            raise InputError("unexpected arm in permutation manifest")
        for order in supplied.values():
            validate_order(order, n)
        orders[qid] = supplied
    if not set(ids) <= set(orders):
        raise InputError("permutation manifest does not cover all cases")
    return questions, orders


def run_one(adapter: Adapter, question: dict, arm: str, order: list[int]) -> dict:
    row = {"model": adapter.config["model"], "question_id": question["id"],
           "arm": arm, "order": order, "candidate_id": None, "correct": False}
    try:
        key = adapter.select(question, order)
        index = keys_for(question).index(key)
        row.update(status="ok", candidate_id=question["candidate_ids"][index],
                   correct=index == question["gold_index"])
    except (ValueError, TypeError, KeyError, IndexError):
        row["status"] = "invalid_selection"
    except (urllib.error.URLError, TimeoutError, OSError):
        row["status"] = "request_failed"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "score"):
        command = sub.add_parser(name)
        command.add_argument("--questions", type=Path, required=True)
        command.add_argument("--cases", type=Path, default=Path("data/cases.jsonl"))
        command.add_argument("--permutations", type=Path, default=Path("data/permutations.jsonl"))
        command.add_argument("--protocol", choices=("all", "orders", "repeats"), default="all")
        command.add_argument("--datasets", nargs="+", choices=("supergpqa", "lexam", "mmlu_pro", "gpqa_diamond"),
                             help="evaluate or score only these datasets")
        command.add_argument("--limit", type=int, help="use the first N selected questions for a small trial")
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--dry-run", action="store_true", help="write request payloads without calling a model")
        else:
            command.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        questions, orders = load_inputs(args.questions, args.cases, args.permutations, args.datasets, args.limit)
        arms = {"all": ALL_ARMS, "orders": ORDER_ARMS, "repeats": REPEAT_ARMS}[args.protocol]
        if args.output.exists():
            raise InputError("output already exists; choose a new path")
        if args.command == "score":
            predictions = read_jsonl(args.predictions)
            if args.datasets or args.limit is not None:
                selected_ids = {q["id"] for q in questions}
                predictions = [row for row in predictions if row["question_id"] in selected_ids]
            result = score(questions, predictions, arms, orders)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"Scored {len(questions)} questions; wrote {args.output}")
            return 0
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        if args.workers < 1:
            raise InputError("workers must be positive")
        if config["api"] == "native" and args.workers != 1:
            raise InputError("native models require workers=1")
        adapter = None if args.dry_run else Adapter(config)
        tasks = [(q, arm, orders[q["id"]][arm]) for q in questions for arm in arms]
        statuses = Counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            if args.dry_run:
                for question, arm, order in tasks:
                    stream.write(wire({"question_id": question["id"], "arm": arm, "order": order,
                                       "payload": payload_for(question, order, config)}) + "\n")
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    records = pool.map(lambda task: run_one(adapter, *task), tasks)
                    for count, record in enumerate(records, 1):
                        stream.write(wire(record) + "\n")
                        stream.flush()
                        statuses[record["status"]] += 1
                        if count % 100 == 0:
                            print(f"Completed {count}/{len(tasks)} requests", file=sys.stderr, flush=True)
        label = "Planned" if args.dry_run else "Evaluated"
        print(f"{label} {len(tasks)} requests for {len(questions)} questions; wrote {args.output}")
        if statuses:
            print("Selections: " + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())))
            if not statuses["ok"]:
                print("No valid selections. Check the model, endpoint, credentials, and configuration.", file=sys.stderr)
                return 2
        return 0
    except InputError as exc:
        print(f"Evaluation stopped: {exc}.", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print("Evaluation stopped: a required file is missing. Check --questions, --config, and the manifest paths.", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Evaluation stopped: invalid JSON at line {exc.lineno}, column {exc.colno}.", file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError, ImportError, AttributeError) as exc:
        # Avoid printing transport exceptions, which may contain credentials or URLs.
        print(f"Evaluation stopped: {type(exc).__name__}. Check inputs and configuration.", file=sys.stderr)
        return 2
