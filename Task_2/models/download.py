# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Lock and download the original Qwen and SigLIP model revisions."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from common.shared import read_json, write_json


def ensure_models(config, paths, components=("vision", "language")):
    lock_path = paths["results"] / "metrics" / "model_revisions.json"
    locks = read_json(lock_path) if lock_path.is_file() else {}
    folders = {}
    for name in ("vision", "language"):
        repo_id = config["models"][name]
        if name not in locks:
            locks[name] = {"repo_id": repo_id, "revision": HfApi().model_info(repo_id).sha}
            write_json(lock_path, locks)
        if locks[name]["repo_id"] != repo_id:
            raise RuntimeError(f"Model lock changed for {name}.")
    for name in components:
        if name not in ("vision", "language"):
            raise ValueError(name)
        repo_id = config["models"][name]
        destination = Path(paths["data"]) / "models" / name
        snapshot_download(repo_id=repo_id, revision=locks[name]["revision"], local_dir=destination,
                          ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.tflite"])
        folders[name] = destination
    return folders, locks
