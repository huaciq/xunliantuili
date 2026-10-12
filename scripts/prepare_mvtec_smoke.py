"""Package a small, real-image MVTec AD subset for platform acceptance testing."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def select_images(folder: Path, count: int) -> list[Path]:
    images = sorted(folder.glob("*.png"))
    if len(images) < count:
        raise ValueError(f"Expected at least {count} PNG images in {folder}; found {len(images)}")
    return images[:count]


def prepare_archive(source: Path, output: Path) -> int:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset root not found: {source}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must have a .zip extension")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    selected: list[Path] = []
    for category in CATEGORIES:
        root = source / category
        selected.extend(select_images(root / "train" / "good", 16))
        selected.extend(select_images(root / "test" / "good", 4))
        test_root = root / "test"
        defects = sorted(
            folder for folder in test_root.iterdir() if folder.is_dir() and folder.name != "good"
        )
        for defect in defects:
            if len(list(defect.glob("*.png"))) < 4:
                continue
            images = select_images(defect, 4)
            masks = [
                root / "ground_truth" / defect.name / f"{image.stem}_mask.png" for image in images
            ]
            missing = [mask for mask in masks if not mask.is_file()]
            if missing:
                raise ValueError(f"Missing corresponding mask: {missing[0]}")
            selected.extend(images)
            selected.extend(masks)
            break
        else:
            raise ValueError(f"No anomaly type with at least 4 PNG images in {test_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, (Path("mvtec_smoke") / path.relative_to(source)).as_posix())
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root containing all 15 MVTec AD categories")
    parser.add_argument("output", type=Path, help="New ZIP file to create")
    args = parser.parse_args()
    try:
        count = prepare_archive(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Created {args.output.resolve()}: 15 categories, {count} files")
    print("Per category: 16 train/good, 4 test/good, 4 test anomalies, 4 masks")
    print("Acceptance subset only; do not use its scores as a research benchmark.")


if __name__ == "__main__":
    main()
