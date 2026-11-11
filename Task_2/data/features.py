# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Cache frozen SigLIP final patch features once and reuse them from RAM."""

import gc
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

from common.shared import read_json, write_json
from data.download import download_images


CHUNK_IMAGES = 256
ENCODER_BATCH = 128


def feature_files(data_root, split):
    return sorted((Path(data_root) / "features" / split).glob("chunk_*.pt"))


def _valid_chunk(path, expected_keys):
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return (payload["keys"] == expected_keys and payload["features"].shape ==
                (len(expected_keys), 196, 768) and payload["features"].dtype == torch.float16)
    except (OSError, KeyError, RuntimeError, ValueError):
        return False


def feature_cache_complete(selection, paths):
    return all(_valid_chunk(Path(paths["data"]) / "features" / split / f"chunk_{index:06d}.pt",
                            keys[index:index + CHUNK_IMAGES])
               for split, keys in selection["image_keys"].items()
               for index in range(0, len(keys), CHUNK_IMAGES))


def cache_features(config, paths, selection, vision_path, chunk_callback=None):
    """Use deterministic image processing; the model output is already post-normalized."""
    all_keys = selection["image_keys"]
    needed = [(split, index, keys[index:index + CHUNK_IMAGES])
              for split, keys in all_keys.items()
              for index in range(0, len(keys), CHUNK_IMAGES)]
    missing = [(split, index, keys) for split, index, keys in needed
               if not _valid_chunk(Path(paths["data"]) / "features" / split / f"chunk_{index:06d}.pt", keys)]
    progress_path = paths["results"] / "metrics" / "feature_cache_progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {"seconds": 0.0, "peak_gpu_gib": 0.0}
    if not missing:
        print("SigLIP feature cache is complete.", flush=True)
        report_path = paths["results"] / "metrics" / "feature_cache.json"
        if not report_path.is_file():
            write_json(report_path, {"images": sum(map(len, all_keys.values())), "remaining_chunks": 0,
                                     "seconds_all_sessions": progress["seconds"],
                                     "peak_gpu_gib_all_sessions": progress["peak_gpu_gib"],
                                     "encoder_batch": ENCODER_BATCH, "feature_dtype": "float16",
                                     "post_layernorm_applied_by_siglip_forward": True})
        return {"images": sum(map(len, all_keys.values())), "remaining_chunks": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("Feature caching requires the Colab GPU.")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(vision_path, dtype=torch.float16,
                                              attn_implementation="sdpa", local_files_only=True).cuda().eval()
    vision.requires_grad_(False)
    last_record = started
    for split, index, keys in missing:
        image_paths = download_images(keys, paths["data"])
        batches = []
        for first in range(0, len(keys), ENCODER_BATCH):
            subset = keys[first:first + ENCODER_BATCH]
            images = []
            for key in subset:
                with Image.open(image_paths[key]) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                features = vision(pixel_values=pixels).last_hidden_state
            if features.shape != (len(subset), 196, 768):
                raise RuntimeError(f"SigLIP patch shape changed: {tuple(features.shape)}")
            batches.append(features.cpu().contiguous().half())
            del pixels, features, images
        destination = Path(paths["data"]) / "features" / split / f"chunk_{index:06d}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        torch.save({"keys": keys, "features": torch.cat(batches)}, temporary)
        temporary.replace(destination)
        now = time.perf_counter()
        progress["seconds"] += now - last_record
        last_record = now
        progress["peak_gpu_gib"] = max(progress["peak_gpu_gib"],
                                       torch.cuda.max_memory_allocated() / 2 ** 30)
        write_json(progress_path, progress)
        if chunk_callback:
            chunk_callback(destination)
        print(f"Cached {split} images {index + 1}-{index + len(keys)} / {len(all_keys[split])}", flush=True)
    progress["seconds"] += time.perf_counter() - last_record
    seconds = progress["seconds"]
    peak = max(progress["peak_gpu_gib"], torch.cuda.max_memory_allocated() / 2 ** 30)
    del vision, processor
    gc.collect()
    torch.cuda.empty_cache()
    report = {"images": sum(map(len, all_keys.values())), "remaining_chunks": 0,
              "seconds_all_sessions": seconds, "peak_gpu_gib_all_sessions": peak,
              "encoder_batch": ENCODER_BATCH, "feature_dtype": "float16",
              "post_layernorm_applied_by_siglip_forward": True}
    write_json(paths["results"] / "metrics" / "feature_cache.json", report)
    return report


class FeatureStore:
    """Keep cached FP16 image features in the 52 GB host RAM during training."""

    def __init__(self, data_root, selection):
        self.chunks = []
        self.index = {}
        for split, keys in selection["image_keys"].items():
            for first in range(0, len(keys), CHUNK_IMAGES):
                path = Path(data_root) / "features" / split / f"chunk_{first:06d}.pt"
                if not _valid_chunk(path, keys[first:first + CHUNK_IMAGES]):
                    raise RuntimeError(f"Missing or invalid SigLIP feature chunk: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                chunk_index = len(self.chunks)
                self.chunks.append(payload["features"])
                for offset, key in enumerate(payload["keys"]):
                    self.index[key] = (chunk_index, offset)

    def get(self, keys, device="cuda"):
        features = torch.stack([self.chunks[chunk][offset] for chunk, offset in
                                (self.index[key] for key in keys)])
        return features.pin_memory().to(device, non_blocking=True) if device.startswith("cuda") else features

    def __len__(self):
        return len(self.index)
