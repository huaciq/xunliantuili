"""Package a point-detection task without modifying its source or split membership."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

WEIGHT_NAME = "mobilenet_v2-b0353104.pth"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(source):
    dataset = source / "dataset200_point"
    manifest = json.loads((source / "manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    split = json.loads(
        (source / "manifests/training_split_160_40.json").read_text(encoding="utf-8")
    )
    listed = set()
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = source.joinpath(*relative.parts)
        if entry["path"] in listed:
            raise ValueError(f"Duplicate manifest entry: {path}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Dataset hash or size mismatch: {path}")
        listed.add(entry["path"])
    image_paths = {
        p.relative_to(source).as_posix()
        for p in (dataset / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }
    if len(image_paths) != manifest["images"]:
        raise ValueError("Image count differs from the source manifest")
    if not {"train", "val"} <= split.keys():
        raise ValueError("Both train and val splits are required")
    all_split = split["train"] + split["val"]
    if not split["train"] or not split["val"]:
        raise ValueError("Train and val splits must not be empty")
    if len(all_split) != len(set(all_split)) or set(all_split) != image_paths:
        raise ValueError("Split must cover every image exactly once without overlap")
    files = []
    stats = {}
    partition_hashes = {}
    for partition in ("train", "val"):
        counts = Counter()
        partition_hashes[partition] = set()
        for relative in split[partition]:
            image = source / relative
            label = dataset / "labels" / (image.stem + ".txt")
            if relative not in listed or label.relative_to(source).as_posix() not in listed:
                raise ValueError(f"Unverified image or label: {relative}")
            tokens = label.read_text(encoding="utf-8").split()
            if tokens:
                if len(tokens) != 3 or tokens[0] not in {"0", "1"}:
                    raise ValueError(f"Expected one class_id x y point in {label}")
                coordinates = [float(value) for value in tokens[1:]]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                    raise ValueError(f"Point coordinates must be in [0, 1]: {label}")
            counts[tokens[0] if tokens else "background"] += 1
            files.extend(
                [
                    (image, f"dataset200_point/{partition}/images/{image.name}"),
                    (label, f"dataset200_point/{partition}/labels/{label.name}"),
                ]
            )
            partition_hashes[partition].add(sha256(image))
        stats[partition] = {"images": len(split[partition]), "classes": dict(counts)}
    if partition_hashes["train"] & partition_hashes["val"]:
        raise ValueError("Identical images occur in both train and validation splits")
    expected = {p.relative_to(source).as_posix() for pair in files for p in [pair[0]]}
    if expected != listed:
        raise ValueError("Source manifest contains files outside the selected image/label pairs")
    return files, {
        "schema_version": 1,
        "seed": split["seed"],
        "counts": stats,
        "source_split": split,
        "source_dataset_manifest": manifest,
    }


def prepare(source, output, weights):
    source, output, weights = source.resolve(), output.resolve(), weights.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    files, manifest = load_plan(source)
    if weights.name != WEIGHT_NAME or not sha256(weights).startswith("b0353104"):
        raise ValueError("Expected the official MobileNetV2 IMAGENET1K_V1 weight")
    trainer = source / "pointModel-batch1.py"
    runner = Path(__file__).with_name("run_point_task.py")
    metadata = {
        "schema_version": 1,
        "trainer_sha256": sha256(trainer),
        "pretrained_sha256": sha256(weights),
        "pretrained_name": WEIGHT_NAME,
        "historical_parameters": json.loads(
            (source / "environment/training_parameters.json").read_text(encoding="utf-8")
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output / "point-dataset-v1.zip", "x", compression=ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("dataset200_point/split_manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(output / "point-code-v1.zip", "x", compression=ZIP_DEFLATED) as archive:
        archive.write(trainer, trainer.name)
        archive.write(runner, "platform_train.py")
        archive.write(weights, f"pretrained/checkpoints/{WEIGHT_NAME}")
        archive.write(source / "requirements-training.txt", "requirements-original.txt")
        archive.writestr("integration.json", json.dumps(metadata, indent=2))
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Created {output / 'point-dataset-v1.zip'}")
    print(f"Created {output / 'point-code-v1.zip'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.weights)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
