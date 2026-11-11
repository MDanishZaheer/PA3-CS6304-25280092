# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Publish checkpoints and feature chunks atomically to mounted Google Drive."""

import hashlib
import os
import shutil
import time
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(source, local_root, drive_root, verify=False):
    source = Path(source)
    relative = source.relative_to(local_root)
    destination = Path(drive_root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part-{time.time_ns()}")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete Drive copy: {relative}")
        digest = _sha256(source) if verify else None
        if verify and _sha256(temporary) != digest:
            raise IOError(f"Drive hash mismatch: {relative}")
        os.replace(temporary, destination)
        if verify:
            destination.with_name(destination.name + ".sha256").write_text(digest + "\n", encoding="ascii")
        print(f"Saved to Drive: {relative}", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def sync_small_results(local_results, drive_results):
    saved = []
    for source in sorted(Path(local_results).rglob("*")):
        if not source.is_file() or source.name.endswith((".sha256", ".tmp")) or ".part-" in source.name:
            continue
        if source.suffix == ".pt":
            continue
        destination = Path(drive_results) / source.relative_to(local_results)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size and (
                destination.stat().st_mtime_ns >= source.stat().st_mtime_ns):
            continue
        saved.append(sync_file(source, local_results, drive_results))
    return saved


def _restore_file(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".restoring")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete restore: {source}")
        hash_path = source.with_name(source.name + ".sha256")
        if hash_path.is_file() and _sha256(temporary) != hash_path.read_text(encoding="ascii").strip():
            raise IOError(f"Saved hash differs for {source}")
        os.replace(temporary, destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def restore_results(local_results, drive_results):
    drive_results = Path(drive_results)
    if not drive_results.is_dir():
        return 0
    checkpoints = sorted((drive_results / "checkpoints").glob("update_*.pt"))
    keep = {checkpoints[-1]} if checkpoints else set()
    restored = 0
    for source in sorted(drive_results.rglob("*")):
        if not source.is_file() or source.name.endswith(".sha256") or ".part-" in source.name:
            continue
        if source.name.startswith("update_") and source.suffix == ".pt" and source not in keep:
            continue
        restored += _restore_file(source, Path(local_results) / source.relative_to(drive_results))
    return restored


def restore_feature_cache(local_data, drive_data):
    drive_features = Path(drive_data) / "features"
    if not drive_features.is_dir():
        return 0
    restored = 0
    for source in sorted(drive_features.rglob("chunk_*.pt")):
        restored += _restore_file(source, Path(local_data) / "features" / source.relative_to(drive_features))
    return restored
