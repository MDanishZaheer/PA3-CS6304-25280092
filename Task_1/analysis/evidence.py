# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build numbered research-question tables from saved experiment outputs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import figures
from evaluation.qualitative import ANNOTATION_FIELDS
from common.shared import CONDITIONS, METHODS, read_configuration, read_json, task_paths, write_json


def _paths():
    return task_paths(read_configuration())["results"]


def _csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError(f"No rows for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _predictions(condition, task):
    path = _paths() / "predictions" / f"{condition}_{task}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return {int(item["doc_id"]): item for item in map(json.loads, stream)}


def _gsm8k_paired_scores():
    """Keep the paired accuracy evidence without committing full generations."""
    path = _paths() / "metrics" / "gsm8k_paired_scores.csv"
    prediction_paths = [_paths() / "predictions" / f"{condition}_gsm8k_cot.jsonl"
                        for condition in CONDITIONS]
    raw_complete = all(source.is_file() for source in prediction_paths)
    if not path.is_file() or (raw_complete and any(source.stat().st_mtime_ns > path.stat().st_mtime_ns
                                                    for source in prediction_paths)):
        records = {condition: _predictions(condition, "gsm8k_cot") for condition in CONDITIONS}
        ids = sorted(records["baseline"])
        if any(sorted(items) != ids for items in records.values()):
            raise ValueError("GSM8K document IDs differ between conditions.")
        _csv(path, [{"doc_id": index, **{condition: records[condition][index]["exact_match"]
                                      for condition in CONDITIONS}} for index in ids])
    rows = _read_csv(path)
    if (len(rows) != 1319 or len({int(row["doc_id"]) for row in rows}) != 1319 or
            any(not all(condition in row for condition in CONDITIONS) for row in rows)):
        raise ValueError("The saved GSM8K paired-score file is incomplete.")
    return {condition: {int(row["doc_id"]): float(row[condition]) for row in rows}
            for condition in CONDITIONS}


def _paired_interval(condition, task, metric, repeats=2000):
    if task == "gsm8k_cot" and metric == "exact_match":
        scores = _gsm8k_paired_scores()
        baseline, comparison = scores["baseline"], scores[condition]
        differences = np.array([comparison[index] - baseline[index]
                                for index in sorted(baseline)])
    else:
        baseline = _predictions("baseline", task)
        comparison = _predictions(condition, task)
        differences = np.array([float(comparison[index][metric]) - float(baseline[index][metric])
                                for index in sorted(baseline)])
    if baseline.keys() != comparison.keys():
        raise ValueError(f"Unpaired {task} document IDs for {condition}.")
    generator = np.random.default_rng(6304)
    bootstraps = np.empty(repeats)
    for iteration in range(repeats):
        bootstraps[iteration] = differences[generator.integers(0, len(differences), len(differences))].mean()
    lower, upper = np.quantile(bootstraps, [0.025, 0.975])
    return float(lower * 100), float(upper * 100)


def build_rq1():
    """Specialization, retention, baseline changes, and paired GSM intervals."""
    root = _paths()
    rows = []
    for condition in CONDITIONS:
        summary = read_json(root / "metrics" / f"{condition}_benchmarks.json")["task_results"]
        gsm = float(summary["gsm8k_cot"]["exact_match,flexible-extract"])
        hella = float(summary["hellaswag"]["acc_norm,none"])
        boolq = float(summary["boolq"]["acc,none"])
        arc = float(summary["arc_easy"]["acc_norm,none"])
        rows.append({"condition": condition, "gsm8k_em": gsm, "hellaswag_acc_norm": hella,
                     "hellaswag_acc": float(summary["hellaswag"]["acc,none"]),
                     "boolq_acc": boolq, "arc_easy_acc_norm": arc,
                     "arc_easy_acc": float(summary["arc_easy"]["acc,none"]),
                     "retention_mean": (hella + boolq + arc) / 3})
    baseline = rows[0]
    for row in rows:
        row["gsm8k_delta_pp"] = (row["gsm8k_em"] - baseline["gsm8k_em"]) * 100
        row["hellaswag_delta_pp"] = (row["hellaswag_acc_norm"] - baseline["hellaswag_acc_norm"]) * 100
        row["boolq_delta_pp"] = (row["boolq_acc"] - baseline["boolq_acc"]) * 100
        row["arc_easy_delta_pp"] = (row["arc_easy_acc_norm"] - baseline["arc_easy_acc_norm"]) * 100
        row["retention_delta_pp"] = (row["retention_mean"] - baseline["retention_mean"]) * 100
        if row["condition"] != "baseline":
            low, high = _paired_interval(row["condition"], "gsm8k_cot", "exact_match")
            row["gsm8k_delta_95ci_low_pp"] = low
            row["gsm8k_delta_95ci_high_pp"] = high
        else:
            row["gsm8k_delta_95ci_low_pp"] = 0.0
            row["gsm8k_delta_95ci_high_pp"] = 0.0
    table = _csv(root / "metrics" / "rq1_benchmark_comparison.csv", rows)
    figure_a = figures.specialization_retention(rows, root / "figures" / "rq1_specialization_retention.png")
    figure_b = figures.benchmark_changes(rows, root / "figures" / "rq1_changes_from_baseline.png")
    write_json(root / "metrics" / "rq1_definitions.json", {
        "retention_mean": "Unweighted mean of HellaSwag acc_norm, BoolQ acc, ARC-Easy acc_norm",
        "gsm8k_delta_95ci": "Paired document bootstrap, 2,000 resamples, seed 6304",
        "scores": "Proportions in CSV; figure axes show percentages and percentage points",
    })
    return {"table": table, "figures": [figure_a, figure_b], "rows": rows}


def build_rq2():
    """Compare full tuning, L2-SP, LoRA rank, and OPLoRA restrictions."""
    root = _paths()
    benchmark = {row["condition"]: row for row in build_rq1()["rows"]}
    rows = []
    for method in METHODS:
        configuration = read_configuration(method)["method"]
        resource = read_json(root / "metrics" / f"{method}_resources.json")
        row = {"method": method, "rank": configuration.get("rank", ""),
               "alpha": configuration.get("alpha", ""), "l2_sp_lambda": configuration.get("l2_sp_lambda", ""),
               "protection_rank": configuration.get("protection_rank", 0),
               "trainable_parameters": resource["trainable_parameters"],
               "gsm8k_em": benchmark[method]["gsm8k_em"],
               "gsm8k_delta_pp": benchmark[method]["gsm8k_delta_pp"],
               "retention_mean": benchmark[method]["retention_mean"],
               "retention_delta_pp": benchmark[method]["retention_delta_pp"]}
        displacement = root / "metrics" / f"{method}_displacement_1250.json"
        row["full_model_l2_displacement"] = read_json(displacement)["l2_displacement"] if displacement.is_file() else ""
        rows.append(row)
    histories = {method: _read_csv(root / "histories" / f"{method}.csv") for method in METHODS}
    if any(len(history) != 1250 for history in histories.values()):
        raise RuntimeError("Each method needs 1,250 completed training-loss rows.")
    table = _csv(root / "metrics" / "rq2_method_restrictions.csv", rows)
    figure = figures.loss_curves(histories, root / "figures" / "rq2_training_loss.png")
    displacement_rows = []
    for method, updates in (("full_ft", (625, 1250)), ("full_ft_l2sp", (1250,))):
        for update in updates:
            value = read_json(root / "metrics" / f"{method}_displacement_{update}.json")
            displacement_rows.append({"method": method, "update": update,
                                      "l2_displacement": value["l2_displacement"]})
    displacement_table = _csv(root / "metrics" / "rq2_full_model_displacement.csv", displacement_rows)
    method_rows = {row["method"]: row for row in rows}
    contrasts = []
    for reference, comparison, label in (
        ("full_ft", "full_ft_l2sp", "L2-SP versus Full FT"),
        ("lora_r4", "lora_r16", "LoRA rank 16 versus rank 4"),
        ("lora_r16", "oplora_r16", "OPLoRA versus LoRA rank 16"),
    ):
        first, second = method_rows[reference], method_rows[comparison]
        contrasts.append({"contrast": label, "reference": reference, "comparison": comparison,
                          "gsm8k_change_pp": 100 * (second["gsm8k_em"] - first["gsm8k_em"]),
                          "retention_mean_change_pp": 100 * (second["retention_mean"] - first["retention_mean"]),
                          "trainable_parameter_change": second["trainable_parameters"] - first["trainable_parameters"]})
    contrast_table = _csv(root / "metrics" / "rq2_controlled_contrasts.csv", contrasts)
    return {"tables": [table, contrast_table, displacement_table], "figures": [figure], "rows": rows}


def build_rq3():
    """Summarize parameter overlap and FP32 fixed-probe response changes."""
    root = _paths()
    diagnostic = read_json(root / "metrics" / "diagnostics.json")
    grouped = defaultdict(list)
    for row in diagnostic["rows"]:
        grouped[(row["condition"], row["layer"], row["dataset"])].append(row)
    rows = []
    for (condition, layer, dataset), records in grouped.items():
        ratios = np.array([item["response_ratio"] for item in records], dtype=float)
        if len(ratios) != 32:
            raise RuntimeError(f"Expected 32 fixed probes for {condition}/{layer}/{dataset}.")
        rows.append({"condition": condition, "layer": layer, "dataset": dataset,
                     "rho": records[0]["rho"], "eta": records[0]["eta"],
                     "response_ratio_mean": float(ratios.mean()),
                     "response_ratio_median": float(np.median(ratios)),
                     "response_ratio_p95": float(np.quantile(ratios, 0.95)), "probe_count": len(ratios)})
    table = _csv(root / "metrics" / "rq3_diagnostic_summary.csv", rows)
    figure = figures.diagnostic_plot(diagnostic["rows"], root / "figures" / "rq3_parameter_activation_diagnostics.png")
    projection = read_json(root / "metrics" / "oplora_r16_resources.json")
    residual = _csv(root / "metrics" / "rq3_projection_residual.csv", [{
        "method": "oplora_r16", "max_left_residual": projection["max_left_residual"],
        "max_right_residual": projection["max_right_residual"]}])
    return {"tables": [table, residual], "figures": [figure], "rows": rows}


def build_rq4():
    """Join quality with trainable count, wall time, VRAM, and saved bytes."""
    root = _paths()
    runtime_path = root / "metrics" / "colab_runtime.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else {}
    benchmark = {row["condition"]: row for row in build_rq1()["rows"]}
    rows = []
    for method in METHODS:
        resource = read_json(root / "metrics" / f"{method}_resources.json")
        evaluation_model = read_json(root / "metrics" / f"{method}_benchmarks.json")["model"]
        checkpoint_bytes = resource["checkpoint_sizes_bytes"].get("1250")
        if checkpoint_bytes is None:
            raise RuntimeError(f"{method} has no recorded update 1250 checkpoint size.")
        rows.append({"method": method,
                     "training_platform": "Colab",
                     "training_gpu": runtime.get("gpu", "unknown"),
                     "trainable_parameters": resource["trainable_parameters"],
                     "trainable_millions": resource["trainable_parameters"] / 1e6,
                     "training_hours": resource["training_seconds"] / 3600,
                     "preprocessing_hours": resource.get("preprocessing_seconds", 0) / 3600,
                     "total_hours": (resource["training_seconds"] + resource.get("preprocessing_seconds", 0)) / 3600,
                     "peak_gpu_gib": resource["peak_gpu_memory_bytes"] / (1024 ** 3),
                     "final_recovery_checkpoint_gib": checkpoint_bytes / (1024 ** 3),
                     "adapter_export_mib": resource.get("adapter_export_bytes", 0) / (1024 ** 2),
                     "basis_storage_mib": resource.get("basis_storage_bytes", 0) / (1024 ** 2),
                     "eval_adapter_preprocessing_hours": evaluation_model.get("adapter_preprocessing_seconds", 0) / 3600,
                     "eval_adapter_merge_seconds": evaluation_model.get("adapter_merge_seconds", 0),
                     "reference_storage_gib": resource.get("reference_storage_bytes", 0) / (1024 ** 3),
                     "optimizer_moment_gib": resource.get("optimizer_moment_bytes", 0) / (1024 ** 3),
                     "cpu_master_gib": resource.get("cpu_master_bytes", 0) / (1024 ** 3),
                     "skipped_update_attempts": resource["skipped_update_attempts"],
                     "gsm8k_em": benchmark[method]["gsm8k_em"],
                     "retention_mean": benchmark[method]["retention_mean"]})
    full = rows[0]
    for row in rows:
        row["trainable_fraction_of_full"] = row["trainable_parameters"] / full["trainable_parameters"]
        row["checkpoint_fraction_of_full"] = row["final_recovery_checkpoint_gib"] / full["final_recovery_checkpoint_gib"]
    table = _csv(root / "metrics" / "rq4_quality_cost.csv", rows)
    figure = figures.cost_plot(rows, root / "figures" / "rq4_quality_cost.png")
    return {"table": table, "figures": [figure], "rows": rows}


def build_rq5():
    """Summarize all 168 outputs and annotated 42 rows when labels are filled."""
    root = _paths()
    generated = []
    for condition in CONDITIONS:
        path = root / "predictions" / f"{condition}_qualitative.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            generated.extend(json.loads(line) for line in stream)
    if len(generated) != 168:
        raise RuntimeError(f"Expected 168 generated responses, got {len(generated)}.")
    grouped = defaultdict(list)
    for item in generated:
        grouped[(item["condition"], item["format"])].append(item)
    output_rows = []
    for (condition, format_name), records in grouped.items():
        lengths = [record["response_tokens"] for record in records]
        output_rows.append({"condition": condition, "format": format_name,
                            "outputs": len(records), "mean_response_tokens": float(np.mean(lengths)),
                            "max_response_tokens": max(lengths),
                            "token_cap_count": sum(record["at_token_cap"] for record in records)})
    output_table = _csv(root / "metrics" / "rq5_output_lengths.csv", output_rows)
    output_figure = figures.qualitative_lengths(output_rows, root / "figures" / "rq5_output_lengths.png")
    reviewed_path = root / "annotations" / "qualitative_42_review.csv"
    annotation_path = reviewed_path if reviewed_path.is_file() else root / "annotations" / "qualitative_42.csv"
    annotations = _read_csv(annotation_path)
    if len(annotations) != 42:
        raise RuntimeError("The qualitative annotation sheet must contain 42 rows.")
    required = ANNOTATION_FIELDS[:-1]
    complete = all(all(str(row[field]).strip() in ("0", "1") for field in required) for row in annotations)
    answer = {"tables": [output_table, annotation_path], "figures": [output_figure],
              "annotations_complete": complete, "pending_annotation_rows": sum(
                  not all(str(row[field]).strip() in ("0", "1") for field in required) for row in annotations)}
    if complete:
        groups = defaultdict(list)
        for row in annotations:
            groups[(row["condition"], row["format"])].append(row)
        summary_rows = []
        for (condition, format_name), records in groups.items():
            summary = {"condition": condition, "format": format_name, "annotated_outputs": len(records)}
            for field in required:
                summary[f"{field}_rate"] = np.mean([int(row[field]) for row in records])
            summary["format_compliance_rate"] = summary.pop("format_compliant_rate")
            summary_rows.append(summary)
        summary_table = _csv(root / "metrics" / "rq5_annotation_summary.csv", summary_rows)
        figure = figures.qualitative_plot(summary_rows, root / "figures" / "rq5_format_compliance.png")
        answer["tables"].append(summary_table)
        answer["figures"].append(figure)
    return answer
