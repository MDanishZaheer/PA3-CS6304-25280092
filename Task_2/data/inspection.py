# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Show annotation-derived labels before looking at model predictions."""

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

from common.shared import read_json, write_json
from data.download import download_image


def build_grounding_label_audit(selection, paths, seed=6304):
    """Choose one seeded pair of each type from train, dev, and test."""
    output = paths["results"] / "figures" / "grounding_label_audit.png"
    manifest_path = paths["results"] / "splits" / "grounding_label_audit.json"
    if output.is_file() and manifest_path.is_file():
        return read_json(manifest_path)
    font_path = paths["root"] / "Task_2" / "assets" / "times.ttf"
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})
    rng = random.Random(seed)
    chosen = []
    for suite in ("grounding_train", "grounding_dev", "grounding_test"):
        for kind in ("presence", "counting"):
            options = [pair for pair in selection[suite] if pair["kind"] == kind]
            chosen.append({"suite": suite, "pair": rng.choice(options)})
    fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 18))
    for row, item in enumerate(chosen):
        pair = item["pair"]
        for position, example in enumerate(pair["examples"]):
            axis = axes[row, position]
            image_path = download_image(example["image_key"], paths["data"])
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.axis("off")
            axis.set_title(f"{item['suite']} / {pair['kind']} / image {position + 1}")
            axis.text(0, -0.05, f"{pair['pair_id']}\n{example['question']}\n"
                      f"COCO-derived label: {example['target']}", transform=axis.transAxes, va="top")
    fig.subplots_adjust(hspace=0.8)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = [{"suite": item["suite"], "pair_id": item["pair"]["pair_id"],
                 "kind": item["pair"]["kind"], "examples": item["pair"]["examples"]}
                for item in chosen]
    write_json(manifest_path, manifest)
    return manifest
