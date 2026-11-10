"""Build three platform ZIPs from frame-named good/bad zipper images."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

DATASETS = ("zipper", "Zipperback", "Zipperfront")
FRAME = re.compile(r"frame_(\d+)(?:_\d+)*", re.IGNORECASE)


def frame(path):
    match = FRAME.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Expected a frame-numbered image name: {path}")
    return int(match[1])


def images(folder):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError(f"No images in {folder}")
    return files


def split_dataset(source, gap=12):
    good, bad = images(source / "good"), images(source / "bad")
    bad_frames = {frame(path) for path in bad}
    good_frames = {path: frame(path) for path in good}
    target = max(2, round(len(good) * 0.2))
    candidates = []
    for cutoff in sorted(set(good_frames.values())):
        train = [
            p for p in good if good_frames[p] < cutoff - gap and good_frames[p] not in bad_frames
        ]
        test = [p for p in good if good_frames[p] >= cutoff]
        if len(train) >= 4 and len(test) >= 2:
            candidates.append((abs(len(test) - target), -len(train), cutoff, train, test))
    if not candidates:
        raise ValueError(f"Cannot split {source.name} without overlapping frames")
    _, _, cutoff, train, test = min(candidates)

    def hashes(paths):
        return {hashlib.sha256(p.read_bytes()).digest() for p in paths}

    if hashes(train) & hashes(test + bad):
        raise ValueError(f"Identical train and test images in {source.name}")
    if hashes(test) & hashes(bad):
        raise ValueError(f"Identical test images have conflicting labels in {source.name}")
    return {"train/good": train, "test/good": test, "test/bad": bad}, {
        "dataset": source.name,
        "normal_holdout_start_frame": cutoff,
        "normal_split_gap_frames": gap,
        "excluded_good": sorted(p.name for p in set(good) - set(train) - set(test)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        plans = [(name, *split_dataset(args.source / name)) for name in DATASETS]
        args.output.mkdir(parents=True, exist_ok=False)
        for name, groups, manifest in plans:
            manifest["split"] = {key: [p.name for p in files] for key, files in groups.items()}
            output = args.output / f"{name}-v1.zip"
            with ZipFile(output, "x", compression=ZIP_STORED) as archive:
                for key, files in groups.items():
                    for path in files:
                        archive.write(path, f"{key}/{path.name}")
                archive.writestr("split_manifest.json", json.dumps(manifest, indent=2))
            print(f"{output}: " + ", ".join(f"{key}={len(v)}" for key, v in groups.items()))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
