# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Stage the two supplied archives from a Kaggle input without republishing raw data."""

import shutil
import zipfile
from pathlib import Path


EXPECTED = {
    "qwen25-math": {"generations.parquet", "probability_metrics.parquet"},
    "qwen25vl-mathvista": {"generations.parquet", "probability_metrics.parquet",
                           "attention_metrics.parquet"},
}


def find_metadata(input_root):
    """Find the supplied MathVista metadata in either input layout."""
    matches = list(Path(input_root).rglob("testmini.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one testmini.json under {input_root}, found {matches}. "
            "Upload the updated Task_3 dataset with input_data/testmini.json.")
    return matches[0]


def stage_archives(input_root):
    """Extract only documented Parquet members to Kaggle's non-output scratch area."""
    input_root = Path(input_root)
    archives = {folder: list(input_root.rglob(f"{folder}-*.zip")) for folder in EXPECTED}
    if all(not matches for matches in archives.values()):
        for folder, filenames in EXPECTED.items():
            found = [path for path in input_root.rglob("generations.parquet") if path.parent.name == folder]
            if len(found) != 1 or not all((found[0].parent / name).is_file() for name in filenames):
                raise FileNotFoundError(f"Attach the supplied {folder} archive or extracted Parquet folder to Kaggle.")
        find_metadata(input_root)
        return input_root
    if any(len(matches) != 1 for matches in archives.values()):
        raise FileNotFoundError(f"Attach exactly one archive per supplied model: {archives}")
    destination = Path("/kaggle/temp/Task_3_data")
    destination.mkdir(parents=True, exist_ok=True)
    for folder, filenames in EXPECTED.items():
        archive = archives[folder][0]
        with zipfile.ZipFile(archive) as source:
            expected_members = {f"{folder}/{name}" for name in filenames}
            actual_members = {name for name in source.namelist() if not name.endswith("/")}
            if actual_members != expected_members:
                raise ValueError(f"Unexpected members in {archive}: {sorted(actual_members ^ expected_members)}")
            for member in sorted(expected_members):
                target = destination / member
                if target.is_file() and target.stat().st_size == source.getinfo(member).file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_stream, target.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
                if target.stat().st_size != source.getinfo(member).file_size:
                    raise IOError(f"Extracted file size mismatch: {target}")
    readmes = list(input_root.rglob("README.MD"))
    if len(readmes) != 1:
        raise FileNotFoundError(f"Attach one supplied README.MD with both archives; found {readmes}")
    shutil.copy2(readmes[0], destination / "README.MD")
    shutil.copy2(find_metadata(input_root), destination / "testmini.json")
    return destination
