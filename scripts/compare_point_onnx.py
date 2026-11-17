"""Compare two point ONNX models on identical RGB inputs and optional point labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_target(path):
    tokens = path.read_text(encoding="utf-8-sig").split()
    if not tokens:
        return None
    if len(tokens) != 3:
        raise ValueError(f"Expected one class_id x y point or an empty label: {path}")
    category, x, y = map(float, tokens)
    if (
        not all(math.isfinite(value) for value in (category, x, y))
        or not category.is_integer()
        or not 0 <= category < 8
        or not 0 <= x <= 1
        or not 0 <= y <= 1
    ):
        raise ValueError(f"Invalid point label: {path}")
    return int(category), x, y


def class_id(output):
    return max(range(8), key=lambda index: output[index + 1])


def validate_output(output):
    if len(output) != 11 or not all(math.isfinite(value) for value in output):
        raise ValueError("Expected 11 finite output values")
    # Report logits/misordered channels without hiding the actual predictions.
    return all(-1e-4 <= value <= 1.0001 for value in output) and math.isclose(
        sum(output[1:9]), 1.0, abs_tol=1e-4
    )


def task_metrics(records, key, threshold):
    tp = fp = tn = fn = class_correct = 0
    distances = []
    positive_confidence, background_confidence = [], []
    for record in records:
        prediction, target = record[key], record["target"]
        detected = prediction[0] >= threshold
        if target is None:
            fp += int(detected)
            tn += int(not detected)
            background_confidence.append(prediction[0])
        else:
            tp += int(detected)
            fn += int(not detected)
            class_correct += int(class_id(prediction) == target[0])
            positive_confidence.append(prediction[0])
            distances.append(math.hypot(prediction[9] - target[1], prediction[10] - target[2]))
    positives, negatives = tp + fn, fp + tn
    return {
        "positive_images": positives,
        "background_images": negatives,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "presence_recall": tp / positives if positives else None,
        "background_false_positive_rate": fp / negatives if negatives else None,
        "class_accuracy_on_all_positives": class_correct / positives if positives else None,
        "mean_point_error_128px_on_all_positives": (
            128 * sum(distances) / positives if positives else None
        ),
        "mean_positive_confidence": (
            sum(positive_confidence) / positives if positives else None
        ),
        "mean_background_confidence": (
            sum(background_confidence) / negatives if negatives else None
        ),
    }


def summarize(records, threshold, atol, rtol):
    if not records:
        raise ValueError("No images to compare")
    differences = [
        abs(a - b)
        for row in records
        for a, b in zip(row["reference"], row["candidate"])
    ]
    jointly_detected = [
        row for row in records
        if row["reference"][0] >= threshold and row["candidate"][0] >= threshold
    ]
    return {
        "samples": len(records),
        "all_outputs_close": all(
            abs(a - b) <= atol + rtol * abs(a)
            for row in records
            for a, b in zip(row["reference"], row["candidate"])
        ),
        "max_abs_diff": max(differences),
        "mean_abs_diff": sum(differences) / len(differences),
        "presence_decisions_changed": sum(
            (row["reference"][0] >= threshold) != (row["candidate"][0] >= threshold)
            for row in records
        ),
        "jointly_detected_images": len(jointly_detected),
        "class_decisions_changed_on_joint_detections": sum(
            class_id(row["reference"]) != class_id(row["candidate"])
            for row in jointly_detected
        ),
        "mean_point_shift_128px_on_joint_detections": (
            128 * sum(
                math.hypot(row["reference"][9] - row["candidate"][9],
                           row["reference"][10] - row["candidate"][10])
                for row in jointly_detected
            ) / len(jointly_detected) if jointly_detected else None
        ),
    }


def open_model(path):
    import onnxruntime as ort

    with path.open("rb") as handle:
        if handle.read(16).startswith(b"FAKE_ONNX"):
            raise ValueError(f"Development placeholder, not an ONNX model: {path}")
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1 or inputs[0].shape != [1, 3, 128, 128]
        or inputs[0].type != "tensor(float)"
        or len(outputs) != 1 or outputs[0].shape != [1, 11]
        or outputs[0].type not in {"tensor(float)", "tensor(double)"}
    ):
        raise ValueError(f"Expected float32 [1,3,128,128] -> float32/float64 [1,11]: {path}")
    return session, {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "input_name": inputs[0].name,
        "output_name": outputs[0].name,
        "input_type": inputs[0].type,
        "output_type": outputs[0].type,
    }


def compare(reference, candidate, images, labels, threshold=0.5, atol=1e-4, rtol=1e-3):
    import numpy as np
    from PIL import Image

    files = sorted(
        path for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported images: {images}")
    sessions, metadata = {}, {}
    for key, path in (("reference", reference), ("candidate", candidate)):
        sessions[key], metadata[key] = open_model(path)
    records = []
    for path in files:
        with Image.open(path) as image:
            # Identical to the original torchvision PIL Resize + ToTensor;
            # Config.MEAN=[0,0,0], Config.STD=[1,1,1] have no further effect.
            rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            value = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / 255
            value = np.ascontiguousarray(value)
        row = {"image": path.name}
        if labels is not None:
            row["target"] = read_target(labels / (path.stem + ".txt"))
        for key, session in sessions.items():
            output = session.run(None, {metadata[key]["input_name"]: value})[0]
            if output.shape != (1, 11):
                raise ValueError(f"Unexpected output shape in {key}: {path.name}")
            row[key] = output[0].tolist()
            row[f"{key}_output_contract_valid"] = validate_output(row[key])
        records.append(row)
    report = {
        "models": metadata,
        "images": str(images.resolve()),
        "labels": str(labels.resolve()) if labels else None,
        "preprocess": "PIL RGB, bilinear resize 128x128, float32 NCHW / 255",
        "output_columns": ["confidence", *[f"class{i}" for i in range(8)], "x", "y"],
        "confidence_threshold": threshold,
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparison": summarize(records, threshold, atol, rtol),
        "output_contract_invalid_samples": {
            key: sum(not row[f"{key}_output_contract_valid"] for row in records)
            for key in sessions
        },
        "predictions": records,
    }
    if labels is not None:
        report["task_metrics"] = {
            key: task_metrics(records, key, threshold) for key in sessions
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path,
                        help="One image directory; use the same independent test/val set for both")
    parser.add_argument("--labels", type=Path,
                        help="Point TXT directory; auto-detected next to images when available")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("Threshold must be finite and in [0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    if args.output.exists():
        parser.error("Output already exists; choose a new report path")
    labels = args.labels
    if labels is None and (args.images.parent / "labels").is_dir():
        labels = args.images.parent / "labels"
    try:
        report = compare(args.reference, args.candidate, args.images, labels,
                         args.threshold, args.atol, args.rtol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"},
                     indent=2, ensure_ascii=False))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
