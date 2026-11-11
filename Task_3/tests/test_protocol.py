# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Small protocol tests that do not load models or assignment exports."""

import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


# Kaggle installs the real packages. Local style/smoke tests can use simple stubs.
try:
    import math_verify  # noqa: F401
except ImportError:
    stub = types.ModuleType("math_verify")
    stub.parse = lambda value, **kwargs: [value]
    stub.verify = lambda left, right, **kwargs: left == right
    sys.modules["math_verify"] = stub
    errors_stub = types.ModuleType("math_verify.errors")
    errors_stub.TimeoutException = type("TimeoutException", (BaseException,), {})
    sys.modules["math_verify.errors"] = errors_stub
try:
    import Levenshtein  # noqa: F401
except ImportError:
    stub = types.ModuleType("Levenshtein")
    stub.distance = lambda left, right: abs(len(left) - len(right)) + sum(a != b for a, b in zip(left, right))
    sys.modules["Levenshtein"] = stub

from data.answers import (NO_ANSWER, _math_equivalent, _parsed_math, answer_groups,
                          canonical_mathvista, comparison_counts, equivalent, last_boxed)
from data.prepare import _merge_export, _rename_aliases, ALIASES, save_splits_and_orders
from cloud.kaggle import (CURRENT_TIMEOUT_PROTOCOL, PRIOR_TIMEOUT_PROTOCOL,
                          find_previous_results, reuse_previous_results)
from methods.adaptive import _simulate
from methods.offline import label_candidates
from methods.selection import candidate_scores, select
from models.live import _question_calls, collect_live, run_worker


class ProtocolTests(unittest.TestCase):
    def test_timeout_update_reuses_only_independent_results(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / "old", Path(folder) / "new"
            manifest = old / "metrics" / "run_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"protocol_hash": "' + PRIOR_TIMEOUT_PROTOCOL + '"}',
                                encoding="utf-8")
            for name in ("math_candidates.parquet", "mathvista_candidates.parquet",
                         "mathvista_labeled.parquet", "math_labeled.parquet"):
                path = old / "predictions" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"saved")
            for name in ("math.json", "mathvista.json"):
                path = old / "splits" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (old / "metrics" / "mathvista_comparison_audit.json").write_text(
                "{}", encoding="utf-8")
            questions = old / "progress" / "live_questions.json"
            questions.parent.mkdir(parents=True)
            questions.write_text("[]", encoding="utf-8")
            for index in range(250):
                path = old / "progress" / "live_calls" / f"{index}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            for index in range(200):
                stem = f"{index * 5:04d}_{(index + 1) * 5:04d}"
                path = old / "progress" / "math_label_chunks" / f"{stem}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                counts = {f"{stage}_{key}": 0 for stage in ("group", "gold")
                          for key in ("calls", "timeouts", "errors", "parse_timeouts")}
                counts.update(group_timeout_questions=0, gold_timeout_questions=0)
                if index == 127:
                    counts["group_errors"] = 1
                if index == 175:
                    counts["group_timeouts"] = 1
                path.write_text(__import__("json").dumps(counts), encoding="utf-8")
                path.with_suffix(".parquet").write_bytes(b"saved")
            with patch("cloud.kaggle._protocol_hash", return_value=PRIOR_TIMEOUT_PROTOCOL):
                status = reuse_previous_results(old, new, CURRENT_TIMEOUT_PROTOCOL)
            self.assertIn("198 MATH chunks", status)
            self.assertFalse((new / "predictions" / "math_labeled.parquet").exists())
            self.assertEqual(len(list((new / "progress" / "math_label_chunks").glob("*.parquet"))), 198)
            self.assertEqual(len(list((new / "progress" / "live_calls").glob("*.json"))), 250)
            clean = new / "progress" / "math_label_chunks" / "0000_0005.json"
            self.assertEqual(__import__("json").loads(clean.read_text())["group_retries"], 0)

    def test_kaggle_ignores_results_inside_code_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder)
            source = inputs / "code" / "Task_3"
            bundled = source / "results" / "metrics" / "run_manifest.json"
            prior = inputs / "previous" / "Task_3" / "results" / "metrics" / "run_manifest.json"
            for path in (bundled, prior):
                path.parent.mkdir(parents=True)
                path.write_text('{"complete": true}', encoding="utf-8")
            self.assertEqual(find_previous_results(inputs, source), [prior.parent.parent])

    def test_balanced_final_box_and_missing_group(self):
        self.assertEqual(last_boxed(r"first \boxed{1} then \boxed{\frac{2}{3}}"), r"\frac{2}{3}")
        self.assertIsNone(last_boxed(r"\boxed{unfinished"))
        self.assertEqual(answer_groups([None, "", "2"], "math", {}),
                         [NO_ANSWER, NO_ANSWER, "answer_0"])

    def test_mathvista_choice_and_precision(self):
        metadata = {"question_type": "multi_choice", "choices": ["97", "102", "107"]}
        self.assertEqual(canonical_mathvista("(B)", metadata), "B")
        self.assertEqual(canonical_mathvista("102", metadata), "B")
        metadata = {"question_type": "free_form", "answer_type": "float", "precision": 2}
        self.assertEqual(canonical_mathvista("1.234", metadata), "1.23")
        self.assertEqual(canonical_mathvista("invalid", metadata), NO_ANSWER)
        self.assertEqual(answer_groups(["invalid", "also invalid"], "mathvista", metadata),
                         [NO_ANSWER, NO_ANSWER])

    def test_math_reference_order_and_bare_latex_environment(self):
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()
        with patch("data.answers.parse", side_effect=lambda expression, **kwargs: [expression]) as parser:
            with patch("data.answers.verify", return_value=True) as verifier:
                self.assertTrue(equivalent(r"\frac{2}{4}", r"\frac{1}{2}", "math", {}))
                verifier.assert_called_once_with([r"$\frac{1}{2}$"], [r"$\frac{2}{4}$"],
                                                 timeout_seconds=120, raise_on_error=True)
                self.assertEqual(parser.call_args_list[0].args[0], r"$\frac{2}{4}$")
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()

    def test_math_verify_timeout_is_counted(self):
        from math_verify.errors import TimeoutException

        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()
        comparison_counts(reset=True)
        with patch("data.answers.parse", side_effect=lambda expression, **kwargs: [expression]):
            with patch("data.answers.verify", side_effect=TimeoutException) as verifier:
                self.assertFalse(equivalent("a", "b", "math", {}))
        self.assertEqual([call.kwargs["timeout_seconds"] for call in verifier.call_args_list],
                         [120])
        self.assertEqual(comparison_counts(reset=True),
                         {"calls": 1, "retries": 0, "timeouts": 1,
                          "errors": 0, "parse_timeouts": 0})
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()

    def test_math_group_uses_shorter_limit(self):
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()
        with patch("data.answers.parse", side_effect=lambda expression, **kwargs: [expression]):
            with patch("data.answers.verify", return_value=True) as verifier:
                self.assertEqual(answer_groups(["a", "b"], "math", {}),
                                 ["answer_0", "answer_0"])
        self.assertEqual(verifier.call_args.kwargs["timeout_seconds"], 15)
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()

    def test_math_verify_succeeds_with_bounded_limit(self):
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()
        comparison_counts(reset=True)
        with patch("data.answers.parse", side_effect=lambda expression, **kwargs: [expression]):
            with patch("data.answers.verify", return_value=True) as verifier:
                self.assertTrue(equivalent("a", "b", "math", {}))
        self.assertEqual(verifier.call_args.kwargs["timeout_seconds"], 120)
        self.assertEqual(comparison_counts(reset=True),
                         {"calls": 1, "retries": 0, "timeouts": 0,
                          "errors": 0, "parse_timeouts": 0})
        _parsed_math.cache_clear()
        _math_equivalent.cache_clear()

    def test_prefix_calibration_and_ties(self):
        first = {"kl_u_p": [1.0, 2.0], "logp": [-1.0, -1.0], "token_count": 2,
                 "group": "answer_0", "rollout_id": "0", "answer": "a", "correct": True}
        second = dict(first, kl_u_p=[100.0, 100.0], group="answer_1", rollout_id="1")
        solo = candidate_scores([first], False)["certainty"][0]
        together = candidate_scores([first, second], False)["certainty"][0]
        self.assertEqual(solo, together)
        winner, _ = select([first, second], "self_consistency")
        self.assertEqual(winner, 0)

    def test_adaptive_first_eligible_prefix(self):
        states = pd.DataFrame([
            {"dataset_name": "math", "split": "evaluation", "sample_id": "x", "order": 0,
             "count": count, "unique_leader": True, "leader": "answer_0",
             "agreement": 0.75, "rwc_agrees": True, "rwc_margin": 0.5,
             "token_cost": count * 10, "correct": count == 32} for count in (4, 8, 16, 32)])
        chosen = _simulate(states, "quality", 0.625, 0.25)
        self.assertEqual(int(chosen.iloc[0]["count"]), 4)
        self.assertEqual(int(_simulate(states, "always32").iloc[0]["count"]), 32)

    def test_split_keeps_all_question_rollouts_together(self):
        frame = pd.DataFrame({"sample_id": np.repeat([str(i) for i in range(1000)], 32),
                              "rollout_id": [str(i) for _ in range(1000) for i in range(32)]})
        with tempfile.TemporaryDirectory() as folder:
            saved = save_splits_and_orders(frame, {"experiment": {"seed": 6304, "orders": 10}}, folder, "math")
            self.assertEqual((len(saved["development"]), len(saved["evaluation"])), (200, 800))
            self.assertEqual(len(saved["orders"][saved["evaluation"][0]]), 10)
            self.assertEqual(len(set(saved["orders"][saved["evaluation"][0]][0])), 32)

    def test_export_join_uses_candidate_keys(self):
        left = pd.DataFrame({"dataset": ["math", "math"], "sample_id": ["q", "q"],
                             "rollout_id": ["0", "1"], "response": ["a", "b"]})
        right = pd.DataFrame({"dataset": ["math", "math"], "sample_id": ["q", "q"],
                              "rollout_id": ["1", "0"], "kl_u_p": [[2.0], [1.0]]})
        joined = _merge_export(left, right, "synthetic")
        self.assertEqual(joined.sort_values("rollout_id")["kl_u_p"].tolist(), [[1.0], [2.0]])

    def test_label_chunks_resume_after_interruption(self):
        frame = pd.DataFrame([
            {"sample_id": str(question), "rollout_id": str(rollout),
             "gold": "1", "answer": "1", "metadata": "{}"}
            for question in range(10) for rollout in range(32)])
        with tempfile.TemporaryDirectory() as folder:
            saved = []
            interrupt = [True]

            def save_pickle(part, path):
                if interrupt[0] and path.name == "0005_0010.parquet":
                    raise InterruptedError("simulated Kaggle interruption")
                path.parent.mkdir(parents=True, exist_ok=True)
                part.to_pickle(path)
                saved.append(path.name)

            with patch("methods.offline.save_frame", side_effect=save_pickle):
                with patch("pandas.read_parquet", side_effect=pd.read_pickle):
                    with self.assertRaises(InterruptedError):
                        label_candidates(frame, "math", folder)
                    self.assertTrue((Path(folder) / "progress" / "math_label_chunks" /
                                     "0000_0005.parquet").is_file())
                    interrupt[0] = False
                    labeled = label_candidates(frame, "math", folder)
            self.assertEqual(len(labeled), len(frame))
            self.assertEqual(saved.count("0000_0005.parquet"), 1)
            self.assertEqual(saved.count("0005_0010.parquet"), 1)

    def test_live_worker_passes_pinned_revision(self):
        from common.shared import save_json

        with tempfile.TemporaryDirectory() as folder:
            questions = Path(folder) / "questions.json"
            save_json(questions, [{"sample_id": "q1", "question": "What is 2 + 2?", "gold": "4"}])
            configuration = {"model": {"revision": "pinned-revision"}}
            with patch("torch.cuda.is_available", return_value=True):
                with patch("models.live._load_model", return_value=(object(), object())):
                    with patch("models.live._question_calls", return_value={}) as calls:
                        run_worker(questions, folder, configuration, 0, 1, float("inf"))
            self.assertEqual(calls.call_args.args[5], "pinned-revision")

    def test_live_calls_are_saved_and_reused(self):
        from common.shared import save_json

        question = {"sample_id": "q1", "question": "What is 2 + 2?", "gold": "4"}
        with tempfile.TemporaryDirectory() as folder:
            questions = Path(folder) / "questions.json"
            save_json(questions, [question])
            response = {"response": r"\boxed{4}", "extracted_answer": "4",
                        "seed": 0, "input_tokens": 10, "output_tokens": 3,
                        "truncated": False, "runtime_seconds": 0.5, "peak_gpu_gb": 1.0}

            def generate(model, tokenizer, messages, cap, seed, device):
                return {**response, "messages": messages, "seed": seed}

            with patch("models.live._generate", side_effect=generate) as generator:
                _question_calls(None, None, question, folder, "cuda:0", "pinned-revision", float("inf"))
                _question_calls(None, None, question, folder, "cuda:0", "pinned-revision", float("inf"))
            self.assertEqual(generator.call_count, 5)
            with patch("common.shared.save_frame", side_effect=lambda frame, path: None):
                live = collect_live(questions, folder, "pinned-revision")
            self.assertEqual(set(live["strategy"]), {"single", "parallel", "refinement"})
            self.assertTrue(live["correct"].all())

    def test_supplied_archive_schema_when_available(self):
        import pyarrow.parquet as parquet

        data_root = Path(__file__).resolve().parents[2] / "Data" / "Task_3"
        archives = list(data_root.glob("*.zip"))
        if len(archives) != 2:
            self.skipTest("Supplied private archives are not present.")
        for archive in archives:
            with zipfile.ZipFile(archive) as source:
                members = [name for name in source.namelist() if name.endswith(".parquet")]
                tables = {}
                for member in members:
                    batch = next(parquet.ParquetFile(source.open(member)).iter_batches(batch_size=1))
                    tables[Path(member).name] = batch.to_pandas()
                attention = tables.get("attention_metrics.parquet")
                if attention is not None:
                    attention = attention.rename(columns={"n_generated_text_tokens": "preceding_token_count"})
                joined = _merge_export(tables["generations.parquet"],
                                       tables["probability_metrics.parquet"], "sample")
                if attention is not None:
                    joined = _merge_export(joined, attention, "sample attention")
                columns = ["response", "answer", "gold", "question", "token_ids", "logp", "kl_u_p"]
                if attention is not None:
                    columns += ["image_attention", "prompt_attention", "generated_attention"]
                joined = _rename_aliases(joined, columns)
                for column in columns:
                    self.assertIn(column, joined)
                self.assertEqual(len(joined.iloc[0]["token_ids"]), len(joined.iloc[0]["logp"]))
                if attention is not None:
                    self.assertEqual(list(joined.iloc[0]["preceding_token_count"]),
                                     list(range(len(joined.iloc[0]["token_ids"]))))


if __name__ == "__main__":
    unittest.main()
