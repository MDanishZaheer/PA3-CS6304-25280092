"""Small checks of Task 2 selection, pair controls, and scoring logic."""

import sys
import tempfile
import unittest
import json
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.answers import consensus, normalize_answer, vqa_soft_score
from data.prepare import _category_counts
from evaluation.evaluate import _bootstrap, _mismatch_mapping
from training.train import TrainingSession
from evaluation.evidence import build_evidence


class Task2ProtocolTests(unittest.TestCase):
    def test_crowd_annotations_block_false_absence_and_ambiguous_counts(self):
        instances = {"annotations": [
            {"image_id": 1, "category_id": 7, "iscrowd": 1},
            {"image_id": 2, "category_id": 7, "iscrowd": 0},
            {"image_id": 2, "category_id": 7, "iscrowd": 0}]}
        counts, present, crowd = _category_counts(instances, {1, 2, 3})
        self.assertEqual(present[7], {1, 2})
        self.assertEqual(crowd[7], {1})
        self.assertEqual(counts[7][2], 2)
        self.assertNotIn(3, present[7])

    def test_vqa_normalization_and_leave_one_out_score(self):
        self.assertEqual(normalize_answer("The two, dogs."), "2 dogs")
        self.assertEqual(normalize_answer("don't"), normalize_answer("dont"))
        refs = ["yes"] * 3 + ["no"] * 7
        self.assertAlmostEqual(vqa_soft_score("yes", refs), 0.9)
        canonical, target, votes = consensus([{"answer": item} for item in refs])
        self.assertEqual((canonical, target, votes), ("no", "no", 7))

    def test_two_caption_passes_cover_each_example_once(self):
        trainer = object.__new__(TrainingSession)
        trainer.config = {"experiment": {"seed": 6304}, "training": {"effective_batch_size": 16}}
        trainer.selection = {"caption_train": list(range(18000))}
        trainer.caption_orders = [np.random.default_rng(6304 + index).permutation(18000)
                                  for index in range(2)]
        for pass_index in range(2):
            presented = [item for update in range(pass_index * 1125, (pass_index + 1) * 1125)
                         for item in trainer._caption_batch(update)]
            self.assertEqual(len(presented), 18000)
            self.assertEqual(set(presented), set(range(18000)))

    def test_mismatches_exclude_original_pair_and_bootstrap_pairs(self):
        pairs = [{"pair_id": f"pair-{index}", "examples": [
            {"image_key": f"val2014:{index * 2 + position:012d}"} for position in (0, 1)]}
            for index in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            mapping = _mismatch_mapping(pairs, path)
            self.assertEqual(mapping, _mismatch_mapping(pairs, path))
            for pair in pairs:
                own = {example["image_key"] for example in pair["examples"]}
                self.assertTrue(all(mapping[f"{pair['pair_id']}:{position}"] not in own
                                    for position in (0, 1)))
        values = {"correct": {}, "complementary": {}}
        for index in range(3):
            values["correct"][f"pair-{index}"] = {"accuracy": 1.0, "reference_nll": 0.5}
            values["complementary"][f"pair-{index}"] = {"accuracy": 0.0, "reference_nll": 1.0}
        summary = _bootstrap(values, 4000, 6304)
        self.assertEqual(summary["correct_advantage_accuracy"]["ci95"], [1.0, 1.0])
        self.assertEqual(summary["correct_advantage_reference_nll"]["ci95"], [0.5, 0.5])

    def test_saved_metrics_rebuild_tables_and_figures_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            for name in ("metrics", "histories", "figures", "examples"):
                (results / name).mkdir(parents=True)
            with (results / "histories" / "training.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["update", "sft_loss"])
                writer.writeheader()
                writer.writerow({"update": 1, "sft_loss": 1.5})
            development = {"update": 0, "caption_reference_loss": 2.0, "suites": {
                name: {"correct_nll": 1.0, "complementary_nll": 1.5,
                       "no_image_nll": 2.0, "image_swap_penalty": 0.5}
                for name in ("grounding_dev", "natural_dev")}}
            (results / "metrics" / "development_0000.json").write_text(json.dumps(development), encoding="utf-8")
            pair_metrics = {name: {"mean": value, "ci95": [value, value]}
                            for name, value in {"accuracy": 0.5, "partner_accuracy": 0.2,
                                "unsupported_answer_rate": 0.3, "pair_accuracy": 0.25,
                                "identical_answer_rate": 0.1, "empty_answer_rate": 0.0,
                                "reference_nll": 1.0, "generation_seconds_per_answer": 0.2}.items()}
            conditions = {name: pair_metrics for name in
                          ("correct", "complementary", "mismatched", "no_image")}
            conditions["correct_advantage_accuracy"] = {"mean": 0.0, "ci95": [0.0, 0.0]}
            conditions["correct_advantage_reference_nll"] = {"mean": 0.0, "ci95": [0.0, 0.0]}
            suite = {"conditions": conditions, "question_type_metrics": {
                name: {"presence": {metric: values["mean"] for metric, values in pair_metrics.items()}}
                for name in ("correct", "complementary", "mismatched", "no_image")}}
            summary = {"suites": {name: suite for name in ("grounding_test", "natural_test")}}
            (results / "metrics" / "final_evaluation.json").write_text(json.dumps(summary), encoding="utf-8")
            (results / "examples" / "qualitative.json").write_text(
                json.dumps({"grounding_test": [], "natural_test": []}), encoding="utf-8")
            paths = {"root": Path(__file__).resolve().parents[2], "results": results,
                     "data": Path(directory) / "data"}
            evidence = build_evidence(paths)
            self.assertEqual(len(evidence["final"]), 8)
            self.assertTrue((results / "figures" / "rq1_curriculum.png").is_file())
            self.assertTrue((results / "metrics" / "question_type_table.csv").is_file())


if __name__ == "__main__":
    unittest.main()
