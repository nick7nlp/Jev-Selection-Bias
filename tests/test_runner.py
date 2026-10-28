"""Offline contract tests with synthetic questions and a local HTTP server."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

from jev_eval.adapters import parse_response, payload_for, validate_config
from jev_eval.protocol import ALL_ARMS, ORDER_ARMS, PROMPT_PREFIX, InputError, decode_answer, input_sha256, task_json
from jev_eval.runner import load_inputs, main, read_jsonl
from jev_eval.scoring import score


def dump_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixtures(directory):
    questions = [{"id": f"synthetic-{i}", "dataset": "synthetic", "state": "",
                  "question": "Select the designated item.", "options": ["oak", "elm", "ash", "fir"],
                  "gold_index": i, "candidate_ids": [f"candidate-{j}" for j in range(4)],
                  "source_labels": ["A", "B", "C", "D"], "option_count": 4} for i in range(2)]
    orders = {arm: ([3, 2, 1, 0] if arm == "reversed" else
                    [1, 3, 0, 2] if arm.startswith("shuffle") else [0, 1, 2, 3]) for arm in ORDER_ARMS}
    dump_jsonl(directory / "questions.jsonl", questions)
    dump_jsonl(directory / "cases.jsonl", [{"id": q["id"]} for q in questions])
    dump_jsonl(directory / "permutations.jsonl", [{"id": q["id"], "orders": orders} for q in questions])
    return questions


@contextmanager
def mock_api(api):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            if api == "jev":
                task = payload
            else:
                prompt = payload["input"] if api == "responses" else payload["messages"][0]["content"]
                task = json.loads(prompt[len(PROMPT_PREFIX):])
            keys = list(task["questions"]["q"]["criteria"])
            text = json.dumps({"choice": keys[0]})
            response = {"model": payload["model"]}
            if api == "jev":
                response["answers"] = {"q": {"type": "choice", "choice": keys[0],
                                            "probabilities": {key: float(key == keys[0]) for key in keys}}}
            elif api == "chat":
                response["choices"] = [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
            elif api == "responses":
                response.update(status="completed", output=[{"type": "message", "role": "assistant", "status": "completed",
                                                            "content": [{"type": "output_text", "text": text}]}])
            else:
                response.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/evaluate", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.questions = fixtures(self.directory)
        self.common = ["--questions", str(self.directory / "questions.jsonl"),
                       "--cases", str(self.directory / "cases.jsonl"),
                       "--permutations", str(self.directory / "permutations.jsonl")]

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()):
            return main(arguments)

    def test_dry_run_without_credentials_preserves_keys_and_hides_gold(self):
        config = self.directory / "config.json"
        config.write_text(json.dumps({"api": "jev", "model": "synthetic-model", "api_key_env": "ABSENT_TEST_KEY"}))
        output = self.directory / "plan.jsonl"
        self.assertEqual(self.invoke(["run", *self.common, "--config", str(config), "--output", str(output), "--dry-run"]), 0)
        rows = read_jsonl(output)
        self.assertEqual(len(rows), 30)
        self.assertEqual(len(set(ALL_ARMS)), 15)
        original, reversed_row = rows[:2]
        self.assertEqual(list(original["payload"]["questions"]["q"]["criteria"]), ["A", "B", "C", "D"])
        self.assertEqual(list(reversed_row["payload"]["questions"]["q"]["criteria"]), ["D", "C", "B", "A"])
        self.assertEqual(reversed_row["payload"]["questions"]["q"]["criteria"]["A"], "oak")
        self.assertNotIn("gold_index", output.read_text())
        self.assertNotIn("candidate_ids", output.read_text())

    def test_dataset_filter_allows_partial_questions_and_full_manifests(self):
        for question, dataset in zip(self.questions, ("lexam", "gpqa_diamond")):
            question["dataset"] = dataset
        dump_jsonl(self.directory / "cases.jsonl", [{"id": q["id"], "dataset": q["dataset"]} for q in self.questions])
        dump_jsonl(self.directory / "questions.jsonl", self.questions[:1])
        questions, orders = load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")), datasets=["lexam"])
        self.assertEqual([q["id"] for q in questions], ["synthetic-0"])
        self.assertEqual(set(orders), {"synthetic-0"})
        with self.assertRaises(ValueError):
            load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")))
        rows = [{"question_id": q["id"], "model": "synthetic-model", "arm": "original",
                 "order": [0, 1, 2, 3], "candidate_id": "candidate-0", "status": "ok"} for q in self.questions]
        dump_jsonl(self.directory / "predictions.jsonl", rows)
        output = self.directory / "filtered-summary.json"
        self.assertEqual(self.invoke(["score", *self.common, "--datasets", "lexam", "--predictions",
                                      str(self.directory / "predictions.jsonl"), "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["question_count"], 1)

    def test_all_http_adapters_end_to_end(self):
        for api in ("jev", "chat", "responses", "anthropic"):
            with self.subTest(api=api), mock_api(api) as (endpoint, received):
                config = self.directory / f"{api}.json"
                config.write_text(json.dumps({"api": api, "model": "synthetic-model", "endpoint": endpoint}))
                output = self.directory / f"{api}.jsonl"
                self.assertEqual(self.invoke(["run", *self.common, "--config", str(config), "--output", str(output), "--workers", "2"]), 0)
                rows = read_jsonl(output)
                self.assertEqual(len(rows), 30)
                self.assertEqual(len(received), 30)
                self.assertTrue(all(row["status"] == "ok" for row in rows))
                self.assertEqual(rows[0]["candidate_id"], "candidate-0")
                self.assertEqual(rows[1]["candidate_id"], "candidate-3")
                summary = self.directory / f"{api}-score.json"
                self.assertEqual(self.invoke(["score", *self.common, "--predictions", str(output), "--output", str(summary)]), 0)
                model = json.loads(summary.read_text())["models"]["synthetic-model"]
                self.assertEqual(model["accuracy"]["original"], 0.5)
                self.assertEqual(model["accuracy"]["reversed"], 0)
                self.assertEqual(model["original_repeats"][-1]["cumulative_candidate_agreement"], 1)
                self.assertEqual(model["reordered"][-1]["cumulative_candidate_agreement"], 0)
                self.assertEqual(model["reordered"][-1]["cumulative_failed_questions"], 2)

    def test_changed_question_or_answer_is_rejected_before_model_initialization(self):
        cases = [{"id": q["id"], "input_sha256": input_sha256(q)} for q in self.questions]
        dump_jsonl(self.directory / "cases.jsonl", cases)
        config = self.directory / "config.json"
        config.write_text(json.dumps({"api": "jev", "model": "synthetic-model"}))
        output = self.directory / "selections.jsonl"
        for field, value in (("question", "Edited question"), ("options", ["cedar", "elm", "ash", "fir"]),
                             ("gold_index", 2)):
            changed = [dict(q) for q in self.questions]
            changed[0][field] = value
            dump_jsonl(self.directory / "questions.jsonl", changed)
            with self.subTest(field=field), patch("jev_eval.runner.Adapter") as adapter, \
                    redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(self.invoke(["run", *self.common, "--config", str(config),
                                              "--output", str(output)]), 2)
                adapter.assert_not_called()
                self.assertIn("reconstruct the input file", errors.getvalue())
                self.assertFalse(output.exists())

    def test_changed_metadata_is_rejected(self):
        cases = [{"id": q["id"], "dataset": q["dataset"], "input_sha256": input_sha256(q)}
                 for q in self.questions]
        dump_jsonl(self.directory / "cases.jsonl", cases)
        self.questions[0]["dataset"] = "changed-dataset"
        dump_jsonl(self.directory / "questions.jsonl", self.questions)
        with self.assertRaisesRegex(InputError, "metadata differs"):
            load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")))

    def test_limit_matches_dry_run_run_and_scoring(self):
        # Reverse the question file to check that the case manifest controls the limit.
        dump_jsonl(self.directory / "questions.jsonl", self.questions[::-1])
        with mock_api("chat") as (endpoint, received):
            config = self.directory / "config.json"
            config.write_text(json.dumps({"api": "chat", "model": "synthetic-model", "endpoint": endpoint}))
            plan, selections, summary = [self.directory / name for name in ("plan.jsonl", "selected.jsonl", "metrics.json")]
            for output, extra in ((plan, ["--dry-run"]), (selections, [])):
                self.assertEqual(self.invoke(["run", *self.common, "--config", str(config), "--limit", "1",
                                              "--output", str(output), *extra]), 0)
                rows = read_jsonl(output)
                self.assertEqual(len(rows), 15)
                self.assertEqual({r["question_id"] for r in rows}, {"synthetic-0"})
            self.assertEqual(len(received), 15)
            self.assertEqual(self.invoke(["score", *self.common, "--limit", "1", "--predictions",
                                          str(selections), "--output", str(summary)]), 0)
            stats = json.loads(summary.read_text())
            self.assertEqual(stats["question_count"], 1)
            self.assertEqual(stats["models"]["synthetic-model"]["accuracy"]["original"], 1)
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(InputError, "positive integer"):
                load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")),
                            limit=limit)

    def test_configuration_errors_are_actionable_without_disclosing_values(self):
        config_path = self.directory / "config.json"
        output = self.directory / "selections.jsonl"
        sensitive = "synthetic-private-value"
        configurations = (
            ({"api": "chat", "model": "synthetic-model", "endpoint": sensitive}, "complete endpoint"),
            ({"api": "chat", "model": "synthetic-model", "endpoint": "https://example.com/" + sensitive,
              "api_key_env": "TEST_MISSING_API_KEY"}, "API key environment variable is unset"),
        )
        for config, expected in configurations:
            config_path.write_text(json.dumps(config))
            with self.subTest(expected=expected), patch.dict(os.environ, {}, clear=True), \
                    redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(self.invoke(["run", *self.common, "--config", str(config_path),
                                              "--output", str(output)]), 2)
                self.assertIn(expected, errors.getvalue())
                self.assertNotIn(sensitive, errors.getvalue())
                self.assertFalse(output.exists())
        for timeout in (0, -1, float("nan"), float("inf"), "120", True):
            with self.subTest(timeout=timeout), self.assertRaises(InputError):
                validate_config({"api": "chat", "model": "synthetic-model", "timeout_seconds": timeout})

    def test_failed_requests_return_nonzero_and_remain_scoreable(self):
        config = self.directory / "config.json"
        config.write_text(json.dumps({"api": "chat", "model": "synthetic-model", "endpoint": "https://example.com"}))
        output = self.directory / "failed.jsonl"
        sensitive = "synthetic-private-transport-detail"
        with patch("jev_eval.adapters.Adapter.select", side_effect=URLError(sensitive)), \
                redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(self.invoke(["run", *self.common, "--config", str(config), "--output", str(output)]), 2)
            self.assertIn("No valid selections", errors.getvalue())
            self.assertNotIn(sensitive, errors.getvalue())
        rows = read_jsonl(output)
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["status"] == "request_failed" for row in rows))
        self.assertNotIn(sensitive, output.read_text())
        summary = self.directory / "metrics.json"
        self.assertEqual(self.invoke(["score", *self.common, "--predictions", str(output), "--output", str(summary)]), 0)
        result = json.loads(summary.read_text())
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["models"]["synthetic-model"]["accuracy"]["original"], 0)

    def test_missing_selections_are_wrong_and_do_not_agree(self):
        questions, orders = load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")))
        rows = [{"question_id": q["id"], "model": "synthetic-model", "arm": arm,
                 "order": orders[q["id"]][arm], "candidate_id": None, "status": "invalid_selection"}
                for q in questions for arm in ALL_ARMS]
        stats = score(questions, rows, ALL_ARMS, orders)["models"]["synthetic-model"]
        self.assertEqual(stats["accuracy"]["original"], 0)
        self.assertEqual(stats["reordered"][-1]["cumulative_failed_questions"], 2)
        self.assertEqual(stats["reordered"][-1]["cumulative_distinct_wrong_candidates"], 0)
        self.assertEqual(stats["reordered"][-1]["cumulative_candidate_agreement"], 0)
        self.assertEqual(stats["reordered"][-1]["complete_selection_questions"], 0)
        # An absent record also remains in the complete question denominator.
        rows.pop()
        stats = score(questions, rows, ALL_ARMS, orders)["models"]["synthetic-model"]
        self.assertEqual(stats["accuracy"]["original_repeat7"], 0)

    def test_agreement_uses_full_question_denominator(self):
        questions, orders = load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")))
        rows = [{"question_id": q["id"], "model": "synthetic-model", "arm": arm,
                 "order": orders[q["id"]][arm], "candidate_id": "candidate-0" if i == 0 else None,
                 "status": "ok" if i == 0 else "invalid_selection"}
                for i, q in enumerate(questions) for arm in ALL_ARMS]
        stats = score(questions, rows, ALL_ARMS, orders)["models"]["synthetic-model"]
        self.assertEqual(stats["original_repeats"][-1]["cumulative_candidate_agreement"], .5)
        self.assertEqual(stats["original_repeats"][-1]["complete_selection_questions"], 1)

    def test_unicode_separator_and_duplicate_option_text_are_preserved(self):
        self.questions[0]["question"] = "A line\u2028separator is part of the question."
        self.questions[0]["options"][1] = self.questions[0]["options"][0]
        with (self.directory / "questions.jsonl").open("w", encoding="utf-8") as stream:
            for question in self.questions:
                stream.write(json.dumps(question, ensure_ascii=False) + "\n")
        loaded, orders = load_inputs(*(self.directory / name for name in ("questions.jsonl", "cases.jsonl", "permutations.jsonl")))
        self.assertEqual(len(loaded), 2)
        self.assertIn("\u2028", loaded[0]["question"])
        self.assertEqual(loaded[0]["options"][:2], ["oak", "oak"])
        self.assertEqual(len(task_json(loaded[0], orders[loaded[0]["id"]]["original"])["questions"]["q"]["criteria"]), 4)
        self.assertEqual(decode_answer('{"choice":"B"}', ["A", "B", "C", "D"], loaded[0]["options"]), "B")

    def test_decoder_accepts_only_unambiguous_answers(self):
        keys = ["A", "B"]
        for text in ('{"choice":"A"}', '```json\n{"choice":"A"}\n```',
                     'Reasoning. {"choice":"A"}', '{"choice":"A"}\n{"choice":"A"}'):
            self.assertEqual(decode_answer(text, keys), "A")
        self.assertEqual(decode_answer('{"choice":"oak"}', keys, ["oak", "elm"]), "A")
        for text in ('A', '{"choice":"A","choice":"B"}', '{"choice":"A"}\n{"choice":"B"}',
                     '{"choice":"C"}', '{"choice":"A","extra":1}', '{"choice":"A"} trailing'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_answer(text, keys)
        with self.assertRaises(ValueError):
            decode_answer('{"choice":"same"}', keys, ["same", "same"])

    def test_truncated_and_refused_chat_are_rejected(self):
        config = {"api": "chat", "model": "synthetic-model"}
        for finish, refusal in (("length", None), ("content_filter", None), ("stop", "refused")):
            with self.subTest(finish=finish), self.assertRaises(ValueError):
                parse_response({"model": config["model"], "choices": [{"message": {
                    "role": "assistant", "content": '{"choice":"A"}', "refusal": refusal}, "finish_reason": finish}]}, self.questions[0], config)

    def test_structured_refusal_tools_and_api_errors_are_rejected(self):
        config = {"api": "chat", "model": "synthetic-model"}
        for signal in ("refusal", "tool_calls", "function_call", "error"):
            raw = {"model": config["model"], "choices": [{"message": {"role": "assistant",
                   "content": '{"choice":"A"}'}, "finish_reason": "stop"}]}
            raw[signal] = {"message": "synthetic signal"}
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                parse_response(raw, self.questions[0], config)

    def test_native_uses_returned_choice_with_rounded_tie(self):
        raw = {"model": "synthetic-model", "answers": {"q": {"type": "choice", "choice": "B",
               "probabilities": {"A": .5, "B": .5, "C": 0, "D": 0}}}}
        config = {"model": "synthetic-model", "api": "jev"}
        self.assertEqual(parse_response(raw, self.questions[0], config), "B")
        raw["answers"]["q"]["choice"] = "D"
        with self.assertRaises(ValueError):
            parse_response(raw, self.questions[0], config)

    def test_all_payloads_exclude_gold(self):
        question = dict(self.questions[0], secret_answer="DO_NOT_SEND")
        for api in ("jev", "chat", "responses", "anthropic", "plugin"):
            payload = payload_for(question, [3, 1, 2, 0], {"api": api, "model": "synthetic-model"})
            encoded = json.dumps(payload)
            self.assertNotIn("DO_NOT_SEND", encoded)
            self.assertNotIn("gold_index", encoded)
        self.assertEqual(task_json(question, [3, 1, 2, 0])["questions"]["q"]["criteria"]["B"], "elm")


if __name__ == "__main__":
    unittest.main()
