"""Check SDK prediction dumps and the compiled A733 NB interface."""

import json
import math
from pathlib import Path
import re
import struct

import numpy as np
import onnxruntime as ort
from PIL import Image


DEFAULT_LIMITS = {
    "max_probability_delta": 0.10,
    "max_coordinate_delta_px": 3.0,
}
FLOAT_ATOL = 0.001
FLOAT_RTOL = 0.001
INPUT_DUMP_SHAPES = {
    "1_3_128_128": False, "3_128_128": False,
    "1_128_128_3": True, "128_128_3": True,
}
OUTPUT_DUMP_SHAPES = {"1_11", "11_1", "11"}


def quality_limits(settings):
    if not isinstance(settings, dict):
        raise ValueError("NB validation_limits must be a JSON object")
    values = DEFAULT_LIMITS | settings
    if values.keys() != DEFAULT_LIMITS.keys():
        raise ValueError("Unknown NB quality limit")
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid NB quality limit: {name}")
    return values


def _tensor_paths(folder, shapes, count):
    files = {}
    suffixes = sorted(shapes, key=len, reverse=True)
    for path in folder.rglob("*.tensor"):
        match = re.match(r"iter_(\d+)_", path.name)
        if not match:
            continue
        shape = next((s for s in suffixes if path.name.endswith(f"_{s}.tensor")), None)
        if shape is None:
            continue
        index = int(match[1])
        if index in files:
            raise ValueError(f"Ambiguous SDK tensor for iteration {index} in {folder}")
        files[index] = (path, shape)
    if set(files) != set(range(count)):
        raise ValueError(
            f"Expected {count} SDK tensors of shapes {suffixes} in {folder}, got {len(files)}"
        )
    return [files[index] for index in range(count)]


def _read_tensor(path, size):
    value = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"Invalid SDK tensor {path}: expected {size} finite values")
    return value


def validate_predictions(source, work, validation, limits):
    count = len(validation)
    float_folder = work / "sanity_float32"
    quantized_folder = work / "sanity_quantized"
    float_inputs = _tensor_paths(float_folder, INPUT_DUMP_SHAPES, count)
    quantized_inputs = _tensor_paths(quantized_folder, INPUT_DUMP_SHAPES, count)
    float_outputs = _tensor_paths(float_folder, OUTPUT_DUMP_SHAPES, count)
    quantized_outputs = _tensor_paths(quantized_folder, OUTPUT_DUMP_SHAPES, count)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    records = []
    for i, case in enumerate(validation):
        with Image.open(work / "validation128" / case["png"]) as image:
            pixels = np.asarray(image, dtype=np.float32)
        value = (pixels.transpose(2, 0, 1)[None] / 255).copy()
        expected = session.run(None, {input_name: value})[0].reshape(-1)
        if expected.size != 11 or not np.isfinite(expected).all():
            raise ValueError(f"Invalid ONNX output for {case['source']}")

        for path, shape in (float_inputs[i], quantized_inputs[i]):
            actual = _read_tensor(path, value.size)
            if INPUT_DUMP_SHAPES[shape]:
                actual = actual.reshape(128, 128, 3).transpose(2, 0, 1)
            if not np.allclose(actual.reshape(value.shape), value, atol=0.0001, rtol=0):
                raise ValueError(f"SDK input differs from the RGB / 255 reference: {path}")

        floating = _read_tensor(float_outputs[i][0], 11)
        quantized = _read_tensor(quantized_outputs[i][0], 11)
        probability_delta = float(np.max(np.abs(quantized[:3] - expected[:3])))
        coordinate_delta = float(np.linalg.norm((quantized[-2:] - expected[-2:]) * 128))
        padding_delta = float(np.max(np.abs(quantized[3:9] - expected[3:9])))
        float_passed = bool(np.allclose(floating, expected, atol=FLOAT_ATOL, rtol=FLOAT_RTOL))
        quantized_passed = (
            probability_delta <= limits["max_probability_delta"]
            and coordinate_delta <= limits["max_coordinate_delta_px"]
            and padding_delta <= 0.01
        )
        records.append({
            "image": case["source"],
            "onnx": expected.tolist(),
            "sdk_float": floating.tolist(),
            "sdk_int8": quantized.tolist(),
            "float_max_abs_delta": float(np.max(np.abs(floating - expected))),
            "int8_probability_delta": probability_delta,
            "int8_coordinate_delta_px": coordinate_delta,
            "int8_padding_delta": padding_delta,
            "confidence_decision_changed_at_0_5": bool((expected[0] >= .5) != (quantized[0] >= .5)),
            "class_decision_changed": bool(np.argmax(expected[1:3]) != np.argmax(quantized[1:3])),
            "float_passed": float_passed,
            "int8_passed": quantized_passed,
        })
    report = {
        "sample_count": count,
        "limits": {**limits, "float_atol": FLOAT_ATOL, "float_rtol": FLOAT_RTOL,
                   "max_padding_delta": 0.01},
        "max_float_abs_delta": max(row["float_max_abs_delta"] for row in records),
        "max_int8_probability_delta": max(row["int8_probability_delta"] for row in records),
        "max_int8_coordinate_delta_px": max(row["int8_coordinate_delta_px"] for row in records),
        "records": records,
    }
    report["status"] = "passed" if all(row["float_passed"] and row["int8_passed"] for row in records) else "failed"
    (work / "quality_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, stage in (("float_passed", "SDK float import"), ("int8_passed", "SDK INT8 quantization")):
        failed = [row for row in records if not row[key]]
        if failed:
            raise ValueError(f"{stage} exceeds error limits on {len(failed)}/{count} validation images; "
                             f"first: {Path(failed[0]['image']).name}; report: {work / 'quality_validation.json'}")
    return {key: value for key, value in report.items() if key != "records"}


def validate_nb_interface(binary, metadata):
    """Read VPMN v2 port records using the layout in the bundled NBInfo tool."""
    inputs = list(metadata["Inputs"].values())
    outputs = list(metadata["Outputs"].values())
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one input and one output in NB metadata")
    inp, out = inputs[0], outputs[0]
    quantize = inp["quantize"]
    if (inp["shape"] != [1, 3, 128, 128] or inp["format"].lower() != "nchw"
            or quantize["qtype"] != "u8" or quantize["zero_point"] != 0
            or not math.isclose(quantize["scale"], 1 / 255, abs_tol=1e-8)
            or out["shape"] != [1, 11] or out["dtype"].lower() != "float16"):
        raise ValueError("NB metadata must describe NCHW UINT8 (scale=1/255, zero point=0) to FLOAT16")

    data = binary.read_bytes()

    def words(count, offset):
        if offset + count * 4 > len(data):
            raise ValueError("Truncated VPMN header or port table")
        return struct.unpack_from(f"<{count}I", data, offset)

    _, version, target, _, size_lo, size_hi, sub_offset, sub_size, sub_count = words(9, 0)
    if (data[:4] != b"VPMN" or version != 0x20000 or target != 0x1000003B
            or (size_lo | (size_hi << 32)) != len(data) or sub_size != 12
            or sub_offset + sub_size * sub_count > len(data)):
        raise ValueError("Invalid A733 VPMN v2 binary header")
    ports = {}
    for i in range(sub_count):
        section_offset, section_size, section_count = words(3, sub_offset + i * sub_size)
        if section_size != 44 or section_offset + section_size * section_count > len(data):
            raise ValueError("Invalid VPMN section table")
        for j in range(section_count):
            kind, off_lo, off_hi, len_lo, len_hi = words(5, section_offset + j * section_size)
            if kind not in (1, 2):
                continue
            offset = off_lo | (off_hi << 32)
            if kind in ports or len_lo != 112 or len_hi or offset + 112 > len(data):
                raise ValueError("Invalid VPMN input or output section")
            fields = words(10, offset)
            scale, zero_point = struct.unpack_from("<fi", data, offset + 40)
            ports[kind] = (list(fields[1:1 + fields[0]]), fields[7], fields[9], scale, zero_point)

    actual_input = ports.get(1)
    actual_output = ports.get(2)
    if (actual_input is None or actual_input[:3] != ([128, 128, 3, 1], 2, 2)
            or actual_input[4] != 0 or not math.isclose(actual_input[3], 1 / 255, abs_tol=1e-8)
            or actual_output is None or actual_output[:3] != ([11, 1], 1, 0)):
        raise ValueError("Compiled NB ports must match UINT8 input and unquantized FLOAT16 output metadata")
    return {"target": hex(target), "input_dtype": "uint8", "input_scale": actual_input[3],
            "input_zero_point": actual_input[4], "output_dtype": "float16", "output_values": 11}
