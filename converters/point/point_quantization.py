"""Checked preparation for point-model per-channel INT8 conversion.

Runs on the training host; SDK operations still run in its existing x86 rootfs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from PIL import Image
import yaml
from sdk_point_entry import quantization_json


RECIPE = "point-pcint8-rgb-u8-v2"


def image_files(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def split_images(onnx_path, dataset):
    """Read the separate calibration and quality-check partitions."""
    manifest = Path(onnx_path).parent / "training_split.json"
    split = json.loads(manifest.read_text(encoding="utf-8"))
    root = Path(dataset).resolve()
    partitions = []
    for key in ("train", "val"):
        paths = split.get(key)
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
            raise ValueError(f"Expected a nonempty {key} image list")
        files = sorted(Path(p).resolve() for p in paths)
        if len(files) != len(set(files)):
            raise ValueError(f"Duplicate image in {key} split")
        for path in files:
            if root not in path.parents or not path.is_file():
                raise ValueError(f"{key} image must exist inside the task dataset: {path}")
        partitions.append(files)
    train, validation = partitions
    if set(train) & set(validation):
        raise ValueError("Overlapping training and validation images")
    return train, validation


def calibration_images(onnx_path, dataset):
    return split_images(onnx_path, dataset)[0]


def resize_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)


def clean_head(source, destination):
    """Remove only Slice(Concat) operations aligned to whole concat inputs.

    Handles both existing server exports (conf/classes/x/y) and newer exports
    (conf/classes/xy), including the server's padded 1..8-class interface.
    No learned operation or partial slice is approximated.
    """
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    shape = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
    if (len(model.graph.input) != 1 or len(model.graph.output) != 1
            or shape(model.graph.input[0]) != [1, 3, 128, 128]
            or shape(model.graph.output[0]) != [1, 11]):
        raise ValueError("Expected point interface [1,3,128,128] -> [1,11]")
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: shape(v) for v in
              list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    replacements = {}
    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 4:
            continue
        concat = producers.get(node.input[0])
        if concat is None or concat.op_type != "Concat":
            continue
        axis = next((a.i for a in concat.attribute if a.name == "axis"), None)
        if axis != 1 or any(name not in constants for name in node.input[1:] if name):
            continue
        start, end, axes = [constants[name].reshape(-1).tolist() for name in node.input[1:4]]
        steps = constants[node.input[4]].reshape(-1).tolist() if len(node.input) > 4 and node.input[4] else [1]
        if len(start) != 1 or len(end) != 1 or axes != [1] or steps != [1] or start[0] < 0:
            continue
        sizes = [shapes.get(name, []) for name in concat.input]
        if any(len(s) != 2 or s[0] != 1 or s[1] < 1 for s in sizes):
            continue
        boundaries = [0]
        for dims in sizes:
            boundaries.append(boundaries[-1] + dims[1])
        stop = min(end[0], boundaries[-1])
        if start[0] not in boundaries or stop not in boundaries or stop <= start[0]:
            continue
        selected = list(concat.input)[boundaries.index(start[0]):boundaries.index(stop)]
        op = "Identity" if len(selected) == 1 else "Concat"
        replacements[node.output[0]] = helper.make_node(
            op, selected, list(node.output), name=(node.name or node.output[0]) + "_direct",
            **({"axis": 1} if op == "Concat" else {}))
    if not replacements:
        raise ValueError("Unrecognized point output head; refusing unchecked conversion")
    for i, node in enumerate(model.graph.node):
        if node.output[0] in replacements:
            model.graph.node[i].CopyFrom(replacements[node.output[0]])
    producers = {name: i for i, node in enumerate(model.graph.node) for name in node.output}
    needed, nodes = set(), set()

    def visit(name):
        if name in needed:
            return
        needed.add(name)
        if name in producers:
            idx = producers[name]
            nodes.add(idx)
            for value in model.graph.node[idx].input:
                visit(value)

    visit(model.graph.output[0].name)
    kept = [n for i, n in enumerate(model.graph.node) if i in nodes]
    initializers = [n for n in model.graph.initializer if n.name in needed]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return {"rewired_slices": len(replacements), "nodes": len(kept)}


def prepare_images(files, work, guest_work, *, validation=False):
    partition = "validation" if validation else "calibration"
    folder = work / f"{partition}128"
    folder.mkdir()
    records, names = [], []
    for i, path in enumerate(files):
        name = f"{i:05d}.png"
        resize_rgb(path).save(folder / name)
        names.append(str(guest_work / folder.name / name))
        records.append({"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "png": name})
    database = "validation.txt" if validation else "dataset.txt"
    (work / database).write_text("".join(n + "\n" for n in names), encoding="utf-8")
    (work / f"{partition}_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def verify_equivalence(source, cleaned, files):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
                for p in (source, cleaned)]
    maximum = 0.0
    for path in files:
        value = np.asarray(resize_rgb(path), dtype=np.float32).transpose(2, 0, 1)[None] / 255
        a, b = [s.run(None, {s.get_inputs()[0].name: value})[0] for s in sessions]
        if a.shape != (1, 11) or b.shape != (1, 11) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite or incompatible point output")
        difference = float(np.abs(a - b).max())
        if difference > 1e-6:
            raise ValueError(f"Output head equivalence check failed: {difference}")
        maximum = max(maximum, difference)
    return {"checked_images": len(files), "max_abs_difference": maximum}


def configure(work, guest_work):
    graph = json.loads((work / "model.json").read_text())["Layers"]
    inputs = [k for k, v in graph.items() if v["op"] == "input"]
    outputs = [k for k, v in graph.items() if v["op"] == "output"]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one SDK input and output")
    port = dict(lid=inputs[0], category="image", dtype="float32", sparse=False,
                tensor_name=None, layout="nchw", shape=[1, 3, 128, 128], fitting="scale",
                preprocess=dict(reverse_channel=False, mean=[0, 0, 0], scale=[1 / 255] * 3,
                    preproc_node_params=dict(add_preproc_node=False, preproc_type="IMAGE_RGB",
                        preproc_image_size=[128, 128], preproc_crop=dict(enable_preproc_crop=False,
                        crop_rect=[0, 0, 128, 128]), preproc_perm=[0, 1, 2, 3])), redirect_to_output=False)
    meta = dict(input_meta=dict(databases=[dict(path=str(guest_work / "dataset.txt"), type="TEXT", ports=[port])]))
    post = dict(postprocess=dict(acuity_postprocs=dict(dump_results=dict(file_type="TENSOR")),
                app_postprocs=[dict(lid=outputs[0], postproc_params=dict(add_postproc_node=False,
                perm=[0, 1], force_float32=False))]))
    (work / "input.yml").write_text(yaml.safe_dump(meta, sort_keys=False))
    meta["input_meta"]["databases"][0]["path"] = str(guest_work / "validation.txt")
    (work / "validation_input.yml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (work / "post.yml").write_text(yaml.safe_dump(post, sort_keys=False))


def keep_input_u8(work):
    graph = json.loads((work / "model.json").read_text())["Layers"]
    path = work / "model.quantize"
    text = path.read_text()
    table = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    linked = {k for k, v in graph.items() if v["op"] == "input"}
    while True:
        expanded = linked | {k for k, v in graph.items() if v["op"] in ("permute", "reshape")
                    and any(p.split(":")[0].lstrip("@") in linked for p in v.get("inputs", []))}
        if expanded == linked:
            break
        linked = expanded
    changed = []
    for lid in sorted(linked):
        key = "@" + lid + ":out0"
        if key in table["quantize_parameters"]:
            table["quantize_parameters"][key] = dict(qtype="u8", quantizer="asymmetric_affine",
                rounding="rtne", max_value=1.0, min_value=0.0, scale=1 / 255, zero_point=0)
            changed.append(key)
    if not changed or not any("@" + lid + ":out0" in changed
                              for lid, v in graph.items() if v["op"] == "input"):
        raise ValueError("Input UINT8 quantization parameter not found")
    (work / "model.quantize.before_input_u8").write_bytes(path.read_bytes())
    path.write_text(quantization_json(table))
    return changed
