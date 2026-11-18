"""Apply the ACUITY layer recommendations after an INT8 accuracy failure."""

import hashlib
import json
import shutil

import yaml


def recommendations(work):
    table = yaml.safe_load((work / "model.quantize").read_text(encoding="utf-8"))
    layers = table.get("customized_quantize_layers", {})
    if not isinstance(layers, dict):
        raise ValueError("SDK customized_quantize_layers must be a mapping")
    graph = json.loads((work / "model.json").read_text(encoding="utf-8"))["Layers"]
    for layer, dtype in layers.items():
        if layer not in graph or not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Invalid SDK hybrid layer recommendation: {layer}")
    return layers


def can_retry(work):
    report = work / "quality_validation.json"
    if not report.is_file():
        return False
    result = json.loads(report.read_text(encoding="utf-8"))
    records = result.get("records", [])
    return bool(
        records
        and all(row["float_passed"] for row in records)
        and any(not row["int8_passed"] for row in records)
        and recommendations(work)
    )


def apply(work, guest_work, sdk_runner, calibration_count):
    layers = recommendations(work)
    if not layers:
        raise ValueError("SDK supplied no hybrid precision recommendations")
    baseline = work / "baseline_int8"
    baseline.mkdir()
    for name in ("model.json", "model.data", "model.quantize", "model.quantize.before_input_u8",
                 "quality_validation.json", "input.yml", "validation_input.yml", "post.yml"):
        shutil.copy2(work / name, baseline / name)
    for name in ("sanity_float32", "sanity_quantized"):
        (work / name).rename(baseline / name)
    (work / "quality_validation.json").unlink()
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in work.glob("*.json")}
    # Hybrid calibration starts with the original SDK table. Reapply the public
    # UINT8 input contract after the SDK has produced the new graph and table.
    shutil.copy2(baseline / "model.quantize.before_input_u8", work / "model.quantize")
    sdk_runner(
        guest_work, False, "quantize", "--model", "model.json", "--model-data", "model.data",
        "--with-input-meta", "input.yml", "--model-quantize", "model.quantize",
        "--quantizer", "perchannel_symmetric_affine", "--qtype", "int8",
        "--device", "CPU", "--batch-size", "1", "--iterations", str(calibration_count),
        "--compute-entropy", "--hybrid",
    )
    candidates = []
    for path in work.glob("*.json"):
        if hashlib.sha256(path.read_bytes()).digest() == before.get(path):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("Layers"), dict):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one new SDK hybrid model graph, found {len(candidates)}")
    graph = candidates[0]
    if not (work / "model.data").is_file() or not (work / "model.quantize").is_file():
        raise ValueError("SDK hybrid model data or quantization table is missing")
    if graph != work / "model.json":
        shutil.copy2(graph, work / "model.json")
    result = {"strategy": "sdk_recommended_hybrid", "recommended_layers": layers,
              "sdk_graph": graph.name, "baseline_directory": str(baseline)}
    (work / "hybrid_quantization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
