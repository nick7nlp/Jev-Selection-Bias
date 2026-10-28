#!/usr/bin/env python3
"""Reconstruct the released evaluation inputs from pinned upstream datasets."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("supergpqa", "lexam", "mmlu_pro", "gpqa_diamond")
INPUT_FIELDS = ("state", "question", "options", "gold_index", "candidate_ids", "source_labels")
GPQA_OPTIONS = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def input_sha256(case: dict) -> str:
    payload = {key: case[key] for key in INPUT_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")


def obtain_source(spec: dict, dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "gpqa_diamond":
        if args.gpqa_csv is None:
            raise ValueError("GPQA Diamond requires --gpqa-csv; see docs/DATA_SOURCES.md")
        path = args.gpqa_csv
    else:
        path = args.source_dir / dataset / spec["path"]
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(f"Missing cached source: {dataset}/{spec['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "selection-bias-data/1.0"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                        while block := response.read(1024 * 1024):
                            handle.write(block)
                verify_file(temporary, spec["sha256"])
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    verify_file(path, spec["sha256"])
    return path


def read_source(path: Path, kind: str) -> list[dict]:
    if kind == "jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Reading source parquet files requires pyarrow") from error
        return pq.read_table(path).to_pylist()
    if kind == "csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported source format: {kind}")


def upstream_id(raw: dict, dataset: str) -> str:
    key = {"supergpqa": "uuid", "lexam": "id", "mmlu_pro": "question_id", "gpqa_diamond": "Record ID"}[dataset]
    return str(raw[key])


def normalize(raw: dict, case: dict) -> dict:
    dataset = case["dataset"]
    uid = case["id"]
    if dataset == "supergpqa":
        question, options = raw["question"], raw["options"]
        gold = ord(raw["answer_letter"]) - ord("A")
        if not 0 <= gold < len(options) or options[gold] != raw["answer"]:
            raise ValueError(f"Inconsistent upstream answer: {uid}")
        subject, difficulty = raw["subfield"], raw["difficulty"]
    elif dataset == "lexam":
        question, options = raw["question"], ast.literal_eval(raw["choices"])
        gold, subject, difficulty = raw["gold"], raw["course"], None
    elif dataset == "mmlu_pro":
        question, options = raw["question"], raw["options"]
        gold, subject, difficulty = raw["answer_index"], raw["category"], None
        if raw["answer"] != chr(65 + gold):
            raise ValueError(f"Inconsistent upstream answer: {uid}")
    elif dataset == "gpqa_diamond":
        initial = case["initial_source_order"]
        if (not isinstance(initial, list) or any(type(i) is not int for i in initial)
                or sorted(initial) != list(range(4))):
            raise ValueError(f"Invalid initial_source_order: {uid}")
        original = [raw[field] for field in GPQA_OPTIONS]
        question, options = raw["Question"], [original[i] for i in initial]
        gold, subject, difficulty = initial.index(0), raw["High-level domain"], None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Empty question: {uid}")
    if not isinstance(options, list) or not all(isinstance(text, str) and text.strip() for text in options):
        raise ValueError(f"Invalid options: {uid}")
    size = len(options)
    if type(gold) is not int or not 0 <= gold < size:
        raise ValueError(f"Invalid gold index: {uid}")
    labels = None if dataset == "lexam" else [chr(65 + i) for i in range(size)]
    candidate_ids = [f"candidate_{i}" for i in range(size)]
    expected = {"option_count": size, "gold_index": gold, "source_labels": labels,
                "candidate_ids": candidate_ids, "subject": subject, "difficulty": difficulty}
    for key, value in expected.items():
        if key in case and case[key] != value:
            raise ValueError(f"Manifest {key} does not match upstream source: {uid}")
    result = {
        "id": uid, "dataset": dataset, "source_id": case["source_id"], "split": case["split"],
        "state": "", "question": question, "options": options, "gold_index": gold,
        "gold_candidate_id": candidate_ids[gold], "candidate_ids": candidate_ids,
        "source_labels": labels, "option_count": size, "subject": subject, "difficulty": difficulty,
    }
    if dataset == "gpqa_diamond":
        result["initial_source_order"] = case["initial_source_order"]
    if dataset == "lexam":
        result["source_config"] = case["source_config"]
    if "input_sha256" in case and case["input_sha256"] != input_sha256(result):
        raise ValueError(f"Reconstructed input checksum mismatch: {uid}")
    return result


def reconstruct(cases: list[dict], sources: dict, args: argparse.Namespace) -> list[dict]:
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate id in cases manifest")
    indexes = {}
    for dataset in sorted({case["dataset"] for case in cases}):
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        configs = {case.get("source_config") for case in cases if case["dataset"] == dataset}
        for spec in sources["datasets"][dataset]["files"]:
            config = spec.get("config")
            if config not in configs:
                continue
            path = obtain_source(spec, dataset, args)
            rows = read_source(path, spec["format"])
            index = {}
            for number, raw in enumerate(rows):
                source_id = upstream_id(raw, dataset)
                if source_id in index:
                    raise ValueError(f"Duplicate upstream ID: {dataset}/{source_id}")
                index[source_id] = (number, raw)
            indexes[(dataset, config)] = index

    output = []
    for case in cases:
        key = (case["dataset"], case.get("source_config"))
        try:
            number, raw = indexes[key][str(case["source_id"])]
        except KeyError as error:
            raise ValueError(f"Source record not found: {case['id']}") from error
        if "source_row" in case and case["source_row"] != number:
            raise ValueError(f"Source row mismatch: {case['id']}")
        output.append(normalize(raw, case))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/cases.jsonl")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/sources.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".cache/sources")
    parser.add_argument("--gpqa-csv", type=Path, help="Official dataset/gpqa_diamond.csv acquired from the publisher")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--output", type=Path, default=ROOT / "data/questions.jsonl")
    parser.add_argument("--offline", action="store_true", help="Use cached source files without downloads")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error("Output already exists; choose another --output or use --force")
    if args.output.resolve() in {args.cases.resolve(), args.sources.resolve()}:
        parser.error("Output must not overwrite a source manifest")
    if args.gpqa_csv is not None and args.output.resolve() == args.gpqa_csv.resolve():
        parser.error("Output must not overwrite the GPQA source")
    with args.cases.open(encoding="utf-8") as handle:
        all_cases = [json.loads(line) for line in handle if line.strip()]
    cases = [case for case in all_cases if case["dataset"] in args.datasets]
    if not cases:
        parser.error("No selected cases")
    if "gpqa_diamond" in {case["dataset"] for case in cases} and args.gpqa_csv is None:
        parser.error("GPQA Diamond requires --gpqa-csv; see docs/DATA_SOURCES.md")
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    output = reconstruct(cases, sources, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"questions": len(output), "datasets": dict(sorted(Counter(row["dataset"] for row in output).items()))}))


if __name__ == "__main__":
    main()
