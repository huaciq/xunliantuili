import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_fixture(root):
    manifest, groups = [], {}
    for split, label in (("train", "0 0.2 0.3"), ("val", "")):
        image = f"dataset200_point/images/{split}.jpg"
        groups[split] = [image]
        for relative, content in (
            (image, split.encode()),
            (f"dataset200_point/labels/{split}.txt", label.encode()),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    (root / "manifests").mkdir()
    (root / "manifests/dataset_manifest.json").write_text(
        json.dumps({"images": 2, "files": manifest})
    )
    split_file = root / "manifests/training_split_160_40.json"
    split_file.write_text(json.dumps({"seed": 42, **groups}))
    return split_file


def test_point_package_preserves_explicit_split_and_background(tmp_path):
    dataset_fixture(tmp_path)
    script = load_script("prepare_point_task")
    files, manifest = script.load_plan(tmp_path)
    assert manifest["counts"] == {
        "train": {"images": 1, "classes": {"0": 1}},
        "val": {"images": 1, "classes": {"background": 1}},
    }
    assert {name for _, name in files} == {
        "dataset200_point/train/images/train.jpg",
        "dataset200_point/train/labels/train.txt",
        "dataset200_point/val/images/val.jpg",
        "dataset200_point/val/labels/val.txt",
    }


def test_point_package_rejects_changed_labels_and_overlapping_split(tmp_path):
    split_file = dataset_fixture(tmp_path)
    script = load_script("prepare_point_task")
    payload = json.loads(split_file.read_text())
    payload["val"] = payload["train"]
    split_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly once"):
        script.load_plan(tmp_path)
    (tmp_path / "dataset200_point/labels/train.txt").write_text("1 0.2 0.3")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        script.load_plan(tmp_path)


def test_point_metrics_keep_logs_and_one_based_steps():
    script = load_script("run_point_task")
    stdout, metrics = io.StringIO(), io.StringIO()
    logger = script.MetricLog(stdout, metrics)
    logger.write("Epoch 1: Train Loss = 0.1234, Val ")
    logger.write("Loss = 0.1000\nSave best model\n")
    logger.write("Epoch 2: Train Loss = 0.0900, Val Loss = 0.0800\n")
    assert "Save best model" in stdout.getvalue()
    rows = list(csv.DictReader(io.StringIO(metrics.getvalue())))
    assert [row["step"] for row in rows] == ["1", "2"]
    assert rows[1]["val/loss"] == "0.08"
