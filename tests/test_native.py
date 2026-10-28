"""Native format and routing tests without model weights or accelerator access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_eval.adapters import Adapter, payload_for, validate_config
from jev_eval.native import bare_prompt, build_native, choice_question, decision_messages


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.question = {"id": "synthetic", "state": "context", "question": "Pick one.",
                         "options": ["oak", "elm", "ash"], "gold_index": 1,
                         "candidate_ids": ["candidate-0", "candidate-1", "candidate-2"],
                         "source_labels": ["A", "B", "C"]}
        self.config = {"api": "native", "model": "synthetic-model", "backend": "jev",
                       "checkpoint_env": "SYNTHETIC_MODEL_PATH"}

    def test_bare_format_uses_positions_and_preserves_candidate_content(self):
        payload = payload_for(self.question, [2, 0, 1], self.config)
        self.assertEqual(list(payload["questions"]["q"]["criteria"]), ["C", "A", "B"])
        self.assertEqual(bare_prompt(payload, list("ABCD")),
                         "[kind] choice\n[state] context\n[question] Pick one.\n[options]\nA) ash\nB) oak\nC) elm\n[decision]:")
        self.assertNotIn("gold_index", json.dumps(payload))

    def test_pplx_format_keeps_original_keys_in_option_descriptions(self):
        payload = payload_for(self.question, [2, 0, 1], self.config)
        messages = decision_messages(payload, list("ABCD"))
        self.assertEqual(messages[1]["content"][0]["text"],
                         "State:\ncontext\n\nQuestion:\nPick one.\n\nOptions:\nA: C: ash\nB: A: oak\nC: B: elm\n\nReturn only the letter code of the best option.")

    def test_native_selection_maps_position_back_to_original_candidate(self):
        def backend(payload):
            keys = list(payload["questions"]["q"]["criteria"])
            return {"q": {"type": "choice", "choice": keys[0],
                          "probabilities": {key: float(i == 0) for i, key in enumerate(keys)}}}

        with patch("jev_eval.native.build_native", return_value=backend):
            adapter = Adapter(self.config)
            self.assertEqual(adapter.select(self.question, [2, 0, 1]), "C")
            self.assertEqual(adapter.select(self.question, [0, 1, 2]), "A")

    def test_excess_candidates_are_rejected_without_dropping_options(self):
        payload = payload_for(self.question, [2, 0, 1], self.config)
        with self.assertRaises(ValueError):
            choice_question(payload, 2)
        self.assertEqual(len(payload["questions"]["q"]["criteria"]), 3)

    def test_native_config_validation_and_local_only_path(self):
        validate_config(self.config)
        for changes in ({"backend": "unknown"}, {"max_input_tokens": 0}, {"parameters": {"temperature": 0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.config, **changes})
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            build_native({"backend": "jev", "checkpoint": str(Path(directory) / "missing")})


if __name__ == "__main__":
    unittest.main()
