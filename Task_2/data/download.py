# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Download only official COCO/VQAv2 metadata and selected COCO images."""

import hashlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from urllib.parse import urlparse

from common.shared import split_image_key


ARCHIVES = {
    "coco_annotations": "https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "vqa_train_annotations": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "vqa_val_annotations": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "vqa_train_questions": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "vqa_val_questions": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "vqa_train_pairs": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "vqa_val_pairs": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}

COCO_2014_ANNOTATIONS_SHA256 = "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009"


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url, destination, attempts=4, sha256=None):
    destination = Path(destination)
    if destination.is_file():
        if sha256 is None or _file_sha256(destination) == sha256:
            return destination
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    # COCO's HTTPS endpoint currently presents a certificate for a different host.
    # Limit the workaround to that host; VQAv2 and model downloads keep TLS checks.
    coco_host = urlparse(url).hostname == "images.cocodataset.org"
    if coco_host:
        urllib3.disable_warnings(InsecureRequestWarning)
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(20, 120), verify=not coco_host) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if block:
                            stream.write(block)
            if sha256 is not None and _file_sha256(temporary) != sha256:
                raise ValueError(f"Checksum mismatch for {url}")
            if destination.suffix.lower() == ".jpg":
                with temporary.open("rb") as stream:
                    if stream.read(3) != b"\xff\xd8\xff":
                        raise ValueError(f"Invalid JPEG response for {url}")
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def ensure_metadata(data_root):
    """Keep annotation archives in ignored Data, extracting only JSON members."""
    raw = Path(data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in ARCHIVES.items():
        sha256 = COCO_2014_ANNOTATIONS_SHA256 if name == "coco_annotations" else None
        archive = download_file(url, raw / f"{name}.zip", sha256=sha256)
        with zipfile.ZipFile(archive) as bundle:
            members = [entry for entry in bundle.infolist() if entry.filename.endswith(".json")]
            if not members:
                raise RuntimeError(f"No JSON in {archive}")
            extracted = []
            for entry in members:
                target = raw / Path(entry.filename).name
                if not target.is_file():
                    with bundle.open(entry) as source, target.open("wb") as sink:
                        while block := source.read(1 << 20):
                            sink.write(block)
                extracted.append(str(target))
        manifest[name] = {"url": url, "files": extracted}
    return manifest


def image_url(key):
    split, image_id = split_image_key(key)
    return f"https://images.cocodataset.org/{split}/COCO_{split}_{image_id:012d}.jpg"


def download_image(key, data_root):
    split, image_id = split_image_key(key)
    path = Path(data_root) / "images" / split / f"COCO_{split}_{image_id:012d}.jpg"
    return download_file(image_url(key), path)


def download_images(keys, data_root, workers=16):
    """Fetch selected images in parallel; never download the full COCO archive."""
    output = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_image, key, data_root): key for key in keys}
        for future in as_completed(futures):
            output[futures[future]] = future.result()
    return output
