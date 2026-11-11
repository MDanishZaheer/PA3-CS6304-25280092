# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Stage Task 3 code and optional previous output in Kaggle's writable area."""

import hashlib
import json
import shutil
from pathlib import Path


PRIOR_TIMEOUT_PROTOCOL = "00d755d27aa99ab1ed23cd6b733202695c5d05ce0cfc7608e49fe3b4abb5c020"
CURRENT_TIMEOUT_PROTOCOL = "48496b34a34dae58874f7cfc65982f48f616ba30151b413539150487a2b4ab29"


def _protocol_hash(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.yaml")]):
        if path.relative_to(root).parts[0] not in (
                "common", "configs", "data", "methods", "models", "evaluation"):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def reuse_previous_results(previous, destination, current_hash):
    """Reuse only outputs independent of the changed MATH equivalence timeout."""
    previous, destination = Path(previous), Path(destination)
    with (previous / "metrics" / "run_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["protocol_hash"] == current_hash:
        shutil.copytree(previous, destination, dirs_exist_ok=True)
        return "exact resume"
    if (manifest["protocol_hash"], current_hash) != (PRIOR_TIMEOUT_PROTOCOL,
                                                       CURRENT_TIMEOUT_PROTOCOL):
        raise ValueError("Previous output uses an unsupported scoring protocol; reuse is unsafe.")
    snapshot = previous / "provenance" / "kaggle_source"
    if _protocol_hash(snapshot) != PRIOR_TIMEOUT_PROTOCOL:
        raise ValueError("Previous output source snapshot does not match its manifest.")
    _copy_file(previous / "metrics" / "run_manifest.json",
               destination / "provenance" / "prior_run_manifest.json")
    retained = ["predictions/math_candidates.parquet",
                "predictions/mathvista_candidates.parquet",
                "predictions/mathvista_labeled.parquet",
                "splits/math.json", "splits/mathvista.json",
                "metrics/mathvista_comparison_audit.json"]
    for name in retained:
        _copy_file(previous / name, destination / name)
    live_root = previous / "progress" / "live_calls"
    live_files = sorted(live_root.glob("*.json"))
    if len(live_files) != 250:
        raise ValueError(f"Expected 250 prior live calls, found {len(live_files)}.")
    for path in live_files:
        _copy_file(path, destination / "progress" / "live_calls" / path.name)
    _copy_file(previous / "progress" / "live_questions.json",
               destination / "progress" / "live_questions.json")
    kept, repeated = 0, []
    for audit in sorted((previous / "progress" / "math_label_chunks").glob("*.json")):
        with audit.open(encoding="utf-8") as stream:
            counts = json.load(stream)
        if any(counts[f"{stage}_{key}"] for stage in ("group", "gold")
               for key in ("timeouts", "errors", "parse_timeouts")):
            repeated.append(audit.stem)
            continue
        chunk = audit.with_suffix(".parquet")
        if not chunk.is_file():
            raise FileNotFoundError(f"Missing prior MATH label chunk: {chunk}")
        target = destination / "progress" / "math_label_chunks"
        _copy_file(chunk, target / chunk.name)
        counts.update(group_retries=0, gold_retries=0)
        (target / audit.name).write_text(json.dumps(counts, indent=2), encoding="utf-8")
        kept += 1
    if kept + len(repeated) != 200:
        raise ValueError("Expected 200 prior MATH label chunks.")
    summary = {"prior_protocol_hash": PRIOR_TIMEOUT_PROTOCOL,
               "current_protocol_hash": current_hash,
               "reused_math_chunks": kept, "recomputed_math_chunks": repeated,
               "reused_live_calls": len(live_files)}
    (destination / "provenance" / "reuse_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return f"checked reuse: {kept} MATH chunks and {len(live_files)} live calls"


def find_previous_results(inputs, source):
    """Find a separately attached previous output, if any."""
    return [path for path in Path(inputs).rglob("results")
            if path.is_dir() and path.parent.name == "Task_3" and
            path.parent != Path(source) and
            (path / "metrics" / "run_manifest.json").is_file()]


def stage_kaggle():
    working = Path("/kaggle/working")
    inputs = Path("/kaggle/input")
    destination = working / "Task_3"
    sources = [path.parent.parent for path in inputs.rglob("base.yaml")
               if path.parent.name == "configs"
               and (path.parent.parent / "input_data" / "README.MD").is_file()]
    if len(sources) != 1:
        raise FileNotFoundError(f"Attach one Task_3 code dataset under Kaggle Add Input; found {sources}.")
    source = sources[0]
    if not (destination / "configs" / "base.yaml").is_file():
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("input_data", "results"))
    previous = find_previous_results(inputs, source)
    if len(previous) > 1:
        raise ValueError(f"Several previous Task 3 outputs are attached: {previous}. Keep only the latest version.")
    bundled = source / "results"
    if previous and (bundled / "metrics" / "run_manifest.json").is_file():
        raise ValueError("Both bundled and separate previous outputs are attached; keep one.")
    selected = previous[0] if previous else bundled
    if (selected / "metrics" / "run_manifest.json").is_file():
        status = reuse_previous_results(selected, destination / "results",
                                        _protocol_hash(destination))
        print(f"Restored Task 3 output from {selected}: {status}")
    return destination, inputs
