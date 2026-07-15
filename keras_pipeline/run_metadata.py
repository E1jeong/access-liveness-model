import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from classes import CLASS_NAMES
from utils import collect_split_items


def make_run_id(model_type):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{model_type}"


def split_hashes(data_dir):
    root = Path(data_dir)
    hashes = {}
    for split in ("train", "validation", "test"):
        items = collect_split_items(data_dir, split)
        payload = [
            (str(Path(rgb).relative_to(root)), str(Path(ir).relative_to(root)), label)
            for rgb, ir, label in items
        ]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        hashes[split] = hashlib.sha256(encoded).hexdigest()
    return hashes


def write_run_metadata(path, run_id, config, data_dir, best_checkpoint, best_metrics):
    metadata = {
        "run_id": run_id,
        "config": config,
        "class_map": CLASS_NAMES,
        "split_hashes": split_hashes(data_dir),
        "best_checkpoint": str(best_checkpoint),
        "best_validation_metrics": best_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
