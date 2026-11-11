# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Download and pin the source assets required for Task 1."""

import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download


def _locked_revision(repo_id, repo_type, lock_path):
    """Reuse a saved Hub commit so reruns cannot silently change inputs."""
    lock_path = Path(lock_path)
    revisions = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {}
    key = f"{repo_type}:{repo_id}"
    if key not in revisions:
        api = HfApi()
        if repo_type == "model":
            revisions[key] = api.model_info(repo_id).sha
        elif repo_type == "dataset":
            revisions[key] = api.dataset_info(repo_id).sha
        else:
            raise ValueError(f"Unsupported repository type: {repo_type}")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = lock_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(lock_path)
    return revisions[key]


def download_training_datasets(data_root, results_root, dataset_configuration):
    """Cache MetaMathQA and GSM8K test using pinned source revisions."""
    data_root = Path(data_root)
    cache_dir = data_root / "huggingface" / "datasets"
    lock_path = Path(results_root) / "metrics" / "hub_revisions.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    metamath_repo = dataset_configuration["repo_id"]
    gsm8k_repo = dataset_configuration["gsm8k_repo_id"]
    metamath_revision = _locked_revision(metamath_repo, "dataset", lock_path)
    gsm8k_revision = _locked_revision(gsm8k_repo, "dataset", lock_path)

    metamath = load_dataset(
        metamath_repo, split="train", revision=metamath_revision, cache_dir=str(cache_dir)
    )
    gsm8k_test = load_dataset(
        gsm8k_repo, "main", split="test", revision=gsm8k_revision, cache_dir=str(cache_dir)
    )
    if not {"query", "original_question", "response"}.issubset(metamath.column_names):
        raise ValueError(f"Unexpected MetaMathQA columns: {metamath.column_names}")
    if not {"question", "answer"}.issubset(gsm8k_test.column_names):
        raise ValueError(f"Unexpected GSM8K columns: {gsm8k_test.column_names}")
    if len(metamath) < dataset_configuration["eligible_examples"]:
        raise ValueError("MetaMathQA has fewer rows than the required training count.")
    if len(gsm8k_test) != 1319:
        raise ValueError(f"Expected 1,319 GSM8K test rows, found {len(gsm8k_test)}.")
    return {
        "metamathqa_rows": len(metamath),
        "gsm8k_test_rows": len(gsm8k_test),
        "metamathqa_revision": metamath_revision,
        "gsm8k_revision": gsm8k_revision,
    }


def download_evaluation_datasets(data_root, results_root, dataset_configurations):
    """Cache the remaining fixed benchmark splits using pinned revisions."""
    cache_dir = Path(data_root) / "huggingface" / "datasets"
    lock_path = Path(results_root) / "metrics" / "hub_revisions.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for configuration in dataset_configurations:
        repo_id = configuration["repo_id"]
        revision = _locked_revision(repo_id, "dataset", lock_path)
        dataset = load_dataset(
            repo_id,
            name=configuration.get("subset"),
            split=configuration["split"],
            revision=revision,
            cache_dir=str(cache_dir),
        )
        expected_rows = configuration["expected_rows"]
        if len(dataset) != expected_rows:
            raise ValueError(
                f"{configuration['task']} should have {expected_rows} rows, found {len(dataset)}."
            )
        downloaded[configuration["task"]] = {"rows": len(dataset), "revision": revision}
    return downloaded


def download_base_model(data_root, results_root, model_repo_id):
    """Download the original Qwen checkpoint to the ignored Data directory."""
    data_root = Path(data_root)
    model_dir = data_root / "models" / "Qwen3-0.6B"
    lock_path = Path(results_root) / "metrics" / "hub_revisions.json"
    revision = _locked_revision(model_repo_id, "model", lock_path)
    snapshot_download(repo_id=model_repo_id, revision=revision, local_dir=str(model_dir))
    return {"model_path": str(model_dir), "model_revision": revision}
