# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Resolve the live model revision from the supplied export documentation."""

import re
from pathlib import Path


def resolve_revision(config, data_root):
    documents = list(Path(data_root).rglob("README.MD")) + list(Path(data_root).rglob("README.md"))
    matches = set()
    for document in documents:
        text = document.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "Qwen/Qwen2.5-3B-Instruct" in line:
                matches.update(re.findall(r"\b[0-9a-f]{40}\b", line, re.IGNORECASE))
    configured = config["model"].get("revision")
    if configured and matches and matches != {configured}:
        raise ValueError(f"Configured model revision {configured} differs from the supplied README.MD: {matches}")
    if configured:
        return configured
    if len(matches) != 1:
        raise ValueError("Cannot identify one pinned Qwen2.5-3B-Instruct revision in the supplied README.MD. Set model.revision in configs/base.yaml to its exact commit hash.")
    revision = next(iter(matches))
    config["model"]["revision"] = revision
    return revision
