# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Copy durable Task 1 results between Colab scratch storage and Drive."""

import hashlib
import os
import shutil
import time
from pathlib import Path

from common.shared import METHODS


CHUNK_BYTES = 8 * 1024 * 1024


def checkpoint_update(method, results_root):
    """Return the largest complete local optimizer update for one method."""
    checkpoint_dir = Path(results_root) / "checkpoints" / method
    updates = []
    for path in checkpoint_dir.glob("update_*.pt"):
        try:
            updates.append(int(path.stem.removeprefix("update_")))
        except ValueError:
            continue
    return max(updates, default=0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(source, results_root, drive_results, verify=False):
    """Publish a complete file; checkpoints also receive a verified hash."""
    source, results_root = Path(source), Path(results_root)
    relative = source.relative_to(results_root)
    destination = Path(drive_results) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part-{time.time_ns()}")
    try:
        digest = hashlib.sha256()
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            for block in iter(lambda: input_stream.read(CHUNK_BYTES), b""):
                output_stream.write(block)
                if verify:
                    digest.update(block)
        if temporary.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete Drive copy: {relative}")
        if verify and _sha256(temporary) != digest.hexdigest():
            raise IOError(f"Drive hash mismatch: {relative}")
        os.replace(temporary, destination)
        if verify:
            hash_path = destination.with_name(destination.name + ".sha256")
            hash_path.write_text(digest.hexdigest() + "\n", encoding="ascii")
        print(f"Saved to Drive: {relative}", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def sync_small_results(results_root, drive_results):
    """Save histories, metrics, predictions, figures, and adapter exports."""
    results_root = Path(results_root)
    saved = []
    for source in sorted(results_root.rglob("*")):
        if not source.is_file() or source.name.endswith((".tmp", ".sha256")):
            continue
        if ".part-" in source.name or (source.name.startswith("update_") and source.suffix == ".pt"):
            continue
        destination = Path(drive_results) / source.relative_to(results_root)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size and (
                destination.stat().st_mtime_ns >= source.stat().st_mtime_ns):
            continue
        saved.append(sync_file(source, results_root, drive_results))
    return saved


def restore_results(results_root, drive_results, checkpoints=True):
    """Restore small files plus only the checkpoints needed to resume and score."""
    results_root, drive_results = Path(results_root), Path(drive_results)
    if not drive_results.is_dir():
        return []
    keep = set()
    if checkpoints:
        for method in METHODS:
            checkpoint_dir = drive_results / "checkpoints" / method
            candidates = []
            for path in checkpoint_dir.glob("update_*.pt"):
                try:
                    candidates.append((int(path.stem.removeprefix("update_")), path))
                except ValueError:
                    continue
            if candidates:
                keep.add(max(candidates)[1])
        midpoint = drive_results / "checkpoints" / "full_ft" / "update_0625.pt"
        if midpoint.is_file():
            keep.add(midpoint)
    restored = []
    for source in sorted(drive_results.rglob("*")):
        if not source.is_file() or ".part-" in source.name or source.name.endswith(".sha256"):
            continue
        if source.name.startswith("update_") and source.suffix == ".pt" and source not in keep:
            continue
        destination = results_root / source.relative_to(drive_results)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".restoring")
        try:
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != source.stat().st_size:
                raise IOError(f"Incomplete restore: {source}")
            hash_path = source.with_name(source.name + ".sha256")
            if hash_path.is_file() and _sha256(temporary) != hash_path.read_text(encoding="ascii").strip():
                raise IOError(f"Checkpoint hash mismatch: {source}")
            os.replace(temporary, destination)
            restored.append(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return restored
