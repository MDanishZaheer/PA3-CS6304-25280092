# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select fixed COCO captions, grounding pairs, and VQAv2 pairs."""

import json
import random
from collections import defaultdict
from pathlib import Path

from common.shared import image_key, read_json, write_json
from data.answers import consensus
from data.download import ensure_metadata


def _metadata_file(raw, name):
    matches = list(Path(raw).rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} under {raw}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _vqa_candidates(raw, split, minimum):
    annotations = _metadata_file(raw, f"v2_mscoco_{split}_annotations.json")["annotations"]
    questions = _metadata_file(raw, f"v2_OpenEnded_mscoco_{split}_questions.json")["questions"]
    pairs = _metadata_file(raw, f"v2_mscoco_{split}_complementary_pairs.json")
    by_annotation = {item["question_id"]: item for item in annotations}
    by_question = {item["question_id"]: item for item in questions}
    selected = defaultdict(list)
    seen_pair_ids = set()
    for first_id, second_id in pairs:
        pair_id = f"vqa-{split}-{min(first_id, second_id)}-{max(first_id, second_id)}"
        if pair_id in seen_pair_ids:
            continue
        seen_pair_ids.add(pair_id)
        if first_id not in by_annotation or second_id not in by_annotation:
            continue
        records = []
        for question_id in (first_id, second_id):
            annotation, question = by_annotation[question_id], by_question[question_id]
            canonical, target, votes = consensus(annotation["answers"])
            records.append({"question_id": question_id, "image_key": image_key(split, question["image_id"]),
                            "question": question["question"], "target": target, "canonical": canonical,
                            "references": [answer["answer"] for answer in annotation["answers"]],
                            "answer_type": annotation["answer_type"], "consensus_votes": votes})
        if (min(item["consensus_votes"] for item in records) < minimum or
                records[0]["canonical"] == records[1]["canonical"] or
                records[0]["question"].strip().lower() != records[1]["question"].strip().lower() or
                records[0]["answer_type"] != records[1]["answer_type"] or
                records[0]["image_key"] == records[1]["image_key"]):
            continue
        answer_type = records[0]["answer_type"]
        selected[answer_type].append({"pair_id": pair_id, "kind": "natural", "answer_type": answer_type,
                                      "examples": records})
    return selected


def _take_natural(candidates, quotas, rng, used_images=None):
    chosen = []
    for answer_type, required in quotas.items():
        available = candidates[answer_type].copy()
        rng.shuffle(available)
        count = 0
        for pair in available:
            keys = {item["image_key"] for item in pair["examples"]}
            if used_images is not None and keys & used_images:
                continue
            chosen.append(pair)
            count += 1
            if used_images is not None:
                used_images.update(keys)
            if count == required:
                break
        if count != required:
            raise RuntimeError(f"Only {count}/{required} eligible {answer_type} VQA pairs.")
    return chosen


def _category_counts(instances, allowed_ids):
    counts = defaultdict(lambda: defaultdict(int))
    present = defaultdict(set)
    crowd = defaultdict(set)
    for annotation in instances["annotations"]:
        image_id = annotation["image_id"]
        if image_id in allowed_ids:
            category = annotation["category_id"]
            present[category].add(image_id)
            if annotation.get("iscrowd", 0):
                crowd[category].add(image_id)
            else:
                counts[category][image_id] += 1
    return counts, present, crowd


def _plural(label):
    exceptions = {"person": "people", "sheep": "sheep", "mouse": "mice", "knife": "knives",
                  "bus": "buses", "toothbrush": "toothbrushes"}
    if label in exceptions:
        return exceptions[label]
    return label[:-1] + "ies" if label.endswith("y") else label + "s"


def _grounding_pool(instances, split, allowed_ids, presence_count, counting_count, rng, used_images=None):
    counts, present, crowd = _category_counts(instances, allowed_ids)
    names = {item["id"]: item["name"] for item in instances["categories"]}
    category_ids = sorted(present)
    ordered_ids = sorted(allowed_ids)
    if not category_ids:
        raise RuntimeError(f"No COCO categories for {split} grounding.")
    generated = {"presence": [], "counting": []}
    seen = set()
    for kind, required in (("presence", presence_count), ("counting", counting_count)):
        attempts = 0
        while len(generated[kind]) < required and attempts < required * 600:
            attempts += 1
            category = rng.choice(category_ids)
            category_counts = counts[category]
            available = [image_id for image_id in ordered_ids
                         if used_images is None or image_key(split, image_id) not in used_images]
            if kind == "presence":
                first = [image_id for image_id in available if image_id in present[category]]
                second = [image_id for image_id in available if image_id not in present[category]]
                question = f"Does this image contain {names[category]}?"
                targets = ("yes", "no")
            else:
                first = [image_id for image_id in available if image_id not in crowd[category]
                         and category_counts.get(image_id, 0) in (1, 2, 3, 4)]
                if not first:
                    continue
                first_id = rng.choice(first)
                second = [image_id for image_id in available
                          if image_id not in crowd[category]
                          and category_counts.get(image_id, 0) in (1, 2, 3, 4)
                          and category_counts[image_id] != category_counts[first_id]]
                first = [first_id]
                question = f"How many {_plural(names[category])} can you see?"
                targets = None
            if not first or not second:
                continue
            first_id, second_id = rng.choice(first), rng.choice(second)
            pair_id = f"ground-{split}-{kind}-{category}-{first_id}-{second_id}"
            if pair_id in seen or first_id == second_id:
                continue
            seen.add(pair_id)
            if kind == "counting":
                targets = (str(category_counts[first_id]), str(category_counts[second_id]))
            examples = [{"image_key": image_key(split, image_id), "question": question,
                         "target": target, "canonical": target}
                        for image_id, target in zip((first_id, second_id), targets)]
            generated[kind].append({"pair_id": pair_id, "kind": kind, "category_id": category,
                                    "examples": examples})
            if used_images is not None:
                used_images.update(item["image_key"] for item in examples)
        if len(generated[kind]) != required:
            raise RuntimeError(f"Only {len(generated[kind])}/{required} {split} {kind} pairs.")
    return generated["presence"] + generated["counting"]


def _caption_examples(captions, split, image_count, per_image, rng, used_images=None):
    by_image = defaultdict(list)
    for item in captions["annotations"]:
        by_image[item["image_id"]].append(item)
    image_ids = sorted(image_id for image_id, items in by_image.items() if len(items) >= per_image)
    rng.shuffle(image_ids)
    chosen = []
    for image_id in image_ids:
        key = image_key(split, image_id)
        if used_images is not None and key in used_images:
            continue
        for item in sorted(by_image[image_id], key=lambda value: value["id"])[:per_image]:
            chosen.append({"example_id": f"caption-{split}-{item['id']}", "image_key": key,
                           "caption": item["caption"]})
        if used_images is not None:
            used_images.add(key)
        if len(chosen) == image_count * per_image:
            break
    if len(chosen) != image_count * per_image:
        raise RuntimeError(f"Only {len(chosen)} caption examples for {split}.")
    return chosen


def prepare_selection(config, paths):
    """Freeze split and example IDs before any model-result inspection."""
    output = paths["results"] / "splits" / "selection.json"
    if output.is_file():
        selection = read_json(output)
        verify_selection(config, selection)
        return selection
    sources = ensure_metadata(paths["data"])
    raw = Path(paths["data"]) / "raw"
    rng = random.Random(config["experiment"]["seed"])
    settings = config["data"]
    train_instances = _metadata_file(raw, "instances_train2014.json")
    val_instances = _metadata_file(raw, "instances_val2014.json")
    train_captions = _metadata_file(raw, "captions_train2014.json")
    val_captions = _metadata_file(raw, "captions_val2014.json")
    train_vqa = _vqa_candidates(raw, "train2014", settings["minimum_consensus"])
    val_vqa = _vqa_candidates(raw, "val2014", settings["minimum_consensus"])
    used_val = set()
    natural_dev = _take_natural(val_vqa, settings["natural_dev_pairs"], rng, used_val)
    natural_test = _take_natural(val_vqa, settings["natural_test_pairs"], rng, used_val)
    natural_train = _take_natural(train_vqa, settings["natural_train_pairs"], rng)
    caption_train = _caption_examples(train_captions, "train2014", settings["caption_train_images"],
                                      settings["captions_per_train_image"], rng)
    train_ids = {int(item["image_key"].split(":")[1]) for item in caption_train}
    val_ids = {item["id"] for item in val_instances["images"]}
    grounding_dev = _grounding_pool(val_instances, "val2014", val_ids, 12, 12, rng, used_val)
    grounding_test = _grounding_pool(val_instances, "val2014", val_ids, 32, 32, rng, used_val)
    caption_dev = _caption_examples(val_captions, "val2014", settings["caption_dev_images"], 1, rng, used_val)
    grounding_train = _grounding_pool(train_instances, "train2014", train_ids, 2048, 2048, rng)
    selection = {"seed": config["experiment"]["seed"], "sources": sources,
                 "caption_train": caption_train, "caption_dev": caption_dev,
                 "grounding_train": grounding_train, "grounding_dev": grounding_dev,
                 "grounding_test": grounding_test, "natural_train": natural_train,
                 "natural_dev": natural_dev, "natural_test": natural_test}
    train_keys = {item["image_key"] for item in caption_train}
    val_keys = {item["image_key"] for item in caption_dev}
    for name, pairs in selection.items():
        if not name.startswith(("grounding_", "natural_")):
            continue
        destination = train_keys if name.endswith("train") else val_keys
        destination.update(example["image_key"] for pair in pairs for example in pair["examples"])
    selection["image_keys"] = {"train2014": sorted(train_keys), "val2014": sorted(val_keys)}
    eval_keys = [set(example["image_key"] for pair in selection[name] for example in pair["examples"])
                 for name in ("natural_dev", "natural_test", "grounding_dev", "grounding_test")]
    eval_keys.append({item["image_key"] for item in caption_dev})
    if any(eval_keys[i] & eval_keys[j] for i in range(len(eval_keys)) for j in range(i + 1, len(eval_keys))):
        raise RuntimeError("Development and final-test images overlap.")
    verify_selection(config, selection)
    write_json(output, selection)
    return selection


def verify_selection(config, selection):
    """Check the fixed quotas, consensus, pair structure, and split isolation."""
    quotas = config["data"]
    if (len(selection["caption_train"]), len(selection["caption_dev"]),
            len(selection["grounding_dev"]), len(selection["grounding_test"]),
            len(selection["natural_train"]), len(selection["natural_dev"]),
            len(selection["natural_test"])) != (18000, 96, 24, 64, 1536, 32, 96):
        raise RuntimeError("Task 2 selected-example quotas changed.")
    for name in ("grounding_train", "grounding_dev", "grounding_test", "natural_train",
                 "natural_dev", "natural_test"):
        for pair in selection[name]:
            if len(pair["examples"]) != 2:
                raise RuntimeError(f"Pair does not contain two examples: {pair['pair_id']}")
            first, second = pair["examples"]
            if (first["image_key"] == second["image_key"] or
                    first["question"] != second["question"] or first["canonical"] == second["canonical"]):
                raise RuntimeError(f"Invalid complementary pair in {name}: {pair['pair_id']}")
            if name.startswith("natural_") and any(example["consensus_votes"] < quotas["minimum_consensus"]
                                                    or len(example["references"]) != 10 for example in (first, second)):
                raise RuntimeError(f"Insufficient VQA human consensus in {pair['pair_id']}")
    train_keys = set(selection["image_keys"]["train2014"])
    if len({item["image_key"] for item in selection["caption_train"]}) != 6000:
        raise RuntimeError("Caption training images are not exactly 6,000 unique images.")
    if len({item["image_key"] for item in selection["caption_dev"]}) != 96:
        raise RuntimeError("Caption development images are not exactly 96 unique images.")
    for name, expected in (("grounding_dev", 12), ("grounding_test", 32)):
        actual = {kind: sum(pair["kind"] == kind for pair in selection[name])
                  for kind in ("presence", "counting")}
        if actual != {"presence": expected, "counting": expected}:
            raise RuntimeError(f"Grounding type quotas changed in {name}: {actual}")
    eval_sets = []
    for name in ("natural_dev", "natural_test", "grounding_dev", "grounding_test"):
        eval_sets.append({example["image_key"] for pair in selection[name] for example in pair["examples"]})
    eval_sets.append({item["image_key"] for item in selection["caption_dev"]})
    if any(key in train_keys for group in eval_sets for key in group):
        raise RuntimeError("Training image appears in development or test.")
    if any(eval_sets[i] & eval_sets[j] for i in range(len(eval_sets)) for j in range(i + 1, len(eval_sets))):
        raise RuntimeError("Development and final-test image sets overlap.")
    for name, expected in (("natural_train", quotas["natural_train_pairs"]),
                           ("natural_dev", quotas["natural_dev_pairs"]),
                           ("natural_test", quotas["natural_test_pairs"])):
        actual = {kind: sum(pair["answer_type"] == kind for pair in selection[name]) for kind in expected}
        if actual != expected:
            raise RuntimeError(f"Natural VQA question-type quotas changed in {name}: {actual}")
    return True
