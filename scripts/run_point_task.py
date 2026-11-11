"""Adapt the original point trainer to the platform's mounted inputs and outputs."""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import runpy
import sys
from pathlib import Path

LOSS_LINE = re.compile(r"^Epoch (\d+): Train Loss = ([0-9.eE+-]+), Val Loss = ([0-9.eE+-]+)$")


class MetricLog:
    def __init__(self, stream, metrics):
        self.stream, self.metrics, self.pending = stream, metrics, ""
        self.writer = csv.writer(metrics)
        self.writer.writerow(["step", "train/loss", "val/loss"])
        self.epochs = []

    def write(self, value):
        self.stream.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = LOSS_LINE.fullmatch(line.strip())
            if match:
                step, train, val = int(match[1]), float(match[2]), float(match[3])
                if not all(math.isfinite(v) for v in (train, val)):
                    raise ValueError("Non-finite loss in training output")
                self.writer.writerow([step, train, val])
                self.metrics.flush()
                self.epochs.append(step)
        return len(value)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def validate_onnx(namespace, config, output):
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from torchvision import transforms

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    onnx_path = output / f"{Path(config.DATA_ROOT).name}.onnx"
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "input"
        or inputs[0].shape != [1, 3, 128, 128]
        or len(outputs) != 1
        or outputs[0].name != "output"
        or outputs[0].shape != [1, 11]
    ):
        raise ValueError("Unexpected ONNX input or output contract")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    base = namespace["MobileNetV2_Point"](config.NUM_CLASSES)
    base.load_state_dict(checkpoint["model_state_dict"])
    reference = namespace["SigmoidConfWrapper"](base, config.NUM_CLASSES).eval()
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    split = json.loads((output / "training_split.json").read_text(encoding="utf-8"))
    predictions = []
    maximum = 0.0
    passed = True
    with torch.inference_mode():
        for filename in split["val"]:
            with Image.open(filename) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            expected = reference(tensor).numpy()
            actual = session.run(["output"], {"input": tensor.numpy()})[0]
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            maximum = max(maximum, float(np.max(np.abs(expected - actual)))) if finite else maximum
            matched = finite and bool(np.allclose(expected, actual, atol=1e-4, rtol=1e-3))
            passed = passed and matched
            predictions.append(
                {
                    "image": Path(filename).relative_to(config.DATA_ROOT).as_posix(),
                    "output": actual.tolist(),
                    "matches_pytorch": matched,
                }
            )
    report = {
        "status": "passed" if passed else "failed",
        "samples": len(predictions),
        "max_abs_diff": maximum,
        "atol": 1e-4,
        "rtol": 1e-3,
        "input": {
            "name": "input",
            "shape": [1, 3, 128, 128],
            "dtype": "float32",
            "preprocess": "RGB, resize 128x128, NCHW, divide by 255",
        },
        "output": {
            "name": "output",
            "shape": [1, 11],
            "columns": [
                "confidence",
                "class0",
                "class1",
                "pad0",
                "pad1",
                "pad2",
                "pad3",
                "pad4",
                "pad5",
                "x",
                "y",
            ],
        },
        "predictions": predictions,
    }
    (output / "onnx_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed or not predictions:
        raise RuntimeError("ONNX validation failed; see onnx_validation.json")
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = Path(os.environ.get("PLATFORM_DATASET_DIR", "/workspace/dataset"))
    parser.add_argument("--data-root", type=Path, default=dataset / "dataset200_point")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("PLATFORM_OUTPUT_DIR", "/workspace/output")),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--cls-weight", type=float, default=0.3)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size) < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.cls_weight)
        or args.cls_weight < 0
    ):
        parser.error("Invalid epochs, batch size, learning rate or class loss weight")
    for partition in ("train", "val"):
        if not (args.data_root / partition / "images").is_dir():
            parser.error("Use the prepared dataset with explicit train/val directories")
    code = Path(__file__).resolve().parent
    metadata = json.loads((code / "integration.json").read_text(encoding="utf-8"))
    trainer = code / "pointModel-batch1.py"
    weight = code / "pretrained/checkpoints" / metadata["pretrained_name"]
    if hashlib.sha256(trainer.read_bytes()).hexdigest() != metadata["trainer_sha256"]:
        raise ValueError("Original trainer checksum mismatch")
    if hashlib.sha256(weight.read_bytes()).hexdigest() != metadata["pretrained_sha256"]:
        raise ValueError("Pretrained weight checksum mismatch")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.hub.set_dir(str(code / "pretrained"))
    namespace = runpy.run_path(str(trainer), run_name="point_trainer")
    config = namespace["Config"]()
    if config.DEVICE.type != "cuda":
        raise RuntimeError("Select one GPU for this task")
    config.DATA_ROOT = str(args.data_root.resolve())
    config.SAVE_DIR = str(args.output.resolve())
    config.BEST_MODEL_PATH = str(args.output.resolve() / "mobilenetv2_point_128.pth")
    config.NUM_CLASSES = 2
    config.EPOCHS, config.BATCH_SIZE = args.epochs, args.batch_size
    config.LR, config.CLS_WEIGHT = args.lr, args.cls_weight
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.data_root / "split_manifest.json"
    if manifest.exists():
        (args.output / "split_manifest.json").write_bytes(manifest.read_bytes())
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as metrics:
        logger = MetricLog(sys.stdout, metrics)
        with contextlib.redirect_stdout(logger):
            print("[stage] train: original trainer, fixed split, two point classes", flush=True)
            namespace["train_model"](config)
            print("[stage] export: original fixed-batch ONNX exporter", flush=True)
            namespace["export_onnx"](config)
            print("[stage] validate: ONNX Runtime versus wrapped PyTorch model", flush=True)
            checkpoint, report = validate_onnx(namespace, config, args.output)
    run = {
        "schema_version": 1,
        "parameters": vars(args),
        "integration": metadata,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "completed_epochs": len(logger.epochs),
        "onnx_validation": report["status"],
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            key: importlib.metadata.version(key)
            for key in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "scikit-learn")
        },
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(run, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"Completed: checkpoint + ONNX, {report['samples']} validation images, "
        f"max_abs_diff={report['max_abs_diff']:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
