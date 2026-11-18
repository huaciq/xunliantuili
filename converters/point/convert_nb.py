"""Convert server point models with the verified RGB / per-channel INT8 recipe."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import onnx

from point_quantization import (RECIPE, calibration_images, clean_head, configure,
                              keep_input_u8, prepare_images, verify_equivalence)


ROOT = Path(__file__).resolve().parent
EMULATION = ROOT.parent / "emulation"
ROOTFS = EMULATION / "rootfs"
PROOT = EMULATION / "proot54_pkg/usr/bin/proot"
QEMU = EMULATION / "bin/qemu-x86_64-static"
PEGASUS = "/usr/local/acuity_command_line_tools/pegasus.py"
VIV_SDK = "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools"
TARGET = "VIP9000NANODI_PLUS_PID0X1000003B"


def within_root(path):
    resolved = Path(path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("Conversion paths must stay inside the training directory")
    return resolved


def run_pegasus(guest_work, export, *args):
    env = {
        "HOME": "/root", "PATH": "/usr/sbin:/usr/bin:/bin",
        "PYTHONPATH": "/usr/local/acuity_command_line_tools",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2",
        "QEMU_CPU": "max", "PROOT_NO_SECCOMP": "1",
        "VSIMULATOR_SHADER_CORE_COUNT": "1", "VSIMULATOR_CONFIG": TARGET,
    }
    if export:
        command = [str(PROOT), "-r", str(ROOTFS), "-w", str(guest_work),
                   "-q", str(QEMU), "/usr/bin/python3.8",
                   str(guest_work / "sdk_point_entry.py"), PEGASUS, *args]
    else:
        command = ["/usr/sbin/chroot", str(ROOTFS),
                   "/usr/bin/qemu-x86_64-static", "/usr/bin/python3.8",
                   str(guest_work / "sdk_point_entry.py"), PEGASUS, *args]
    print("SDK stage:", args[0], flush=True)
    subprocess.run(command, env=env, check=True)


def main(onnx_path, dataset_path, nb_path, *, workspace_root=None, sdk_runner=None):
    onnx_path = within_root(onnx_path)
    dataset_path = within_root(dataset_path)
    nb_path = within_root(nb_path)
    if not onnx_path.is_file() or (workspace_root is None and not all(path.is_file() for path in (PROOT, QEMU))):
        raise FileNotFoundError("ONNX model or isolated converter is missing")
    if nb_path.exists():
        raise FileExistsError("Refusing to replace an existing model artifact")
    files = calibration_images(onnx_path, dataset_path)
    task_id = hashlib.sha256(str(nb_path).encode()).hexdigest()[:16] + "-" + uuid.uuid4().hex[:12]
    guest_work = Path("/workspace/training/converter-jobs") / task_id
    # SDK storage is selected by the trusted deployment backend. Model, dataset
    # and output paths above remain confined to ROOT for all HTTP callers.
    backend_root = ROOTFS if workspace_root is None else Path(workspace_root).resolve()
    run_sdk = run_pegasus if sdk_runner is None else sdk_runner
    work = backend_root / guest_work.relative_to("/")
    work.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "sdk_point_entry.py", work / "sdk_point_entry.py")
    report = {"recipe": RECIPE, "workspace": str(guest_work), "host_workspace": str(work), "calibration_count": len(files),
              "quantizer": "perchannel_symmetric_affine", "qtype": "int8",
              "quantization_table_io": "json-yaml-subset",
              "input": {"shape": [1, 3, 128, 128], "qtype": "u8", "scale": 1 / 255, "zero_point": 0},
              "source_onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()}
    cleaned = work / "model.onnx"
    report["head"] = clean_head(onnx_path, cleaned)
    report["equivalence"] = verify_equivalence(onnx_path, cleaned, files)
    prepare_images(files, work, guest_work)
    print(json.dumps(report, indent=2), flush=True)
    model = onnx.load(str(cleaned))
    run_sdk(guest_work, False, "import", "onnx", "--model", "model.onnx",
                "--output-model", "model.json", "--output-data", "model.data",
                "--inputs", model.graph.input[0].name,
                "--input-size-list", "1,3,128,128", "--size-with-batch", "true",
                "--outputs", model.graph.output[0].name)
    configure(work, guest_work)
    run_sdk(guest_work, False, "quantize", "--model", "model.json", "--model-data", "model.data",
                "--with-input-meta", "input.yml", "--model-quantize", "model.quantize",
                "--quantizer", "perchannel_symmetric_affine", "--qtype", "int8",
                "--device", "CPU", "--batch-size", "1", "--iterations", str(len(files)),
                "--compute-entropy", "--rebuild")
    report["uint8_ports"] = keep_input_u8(work)
    for dtype in ("float32", "quantized"):
        quantize = ["--model-quantize", "model.quantize"] if dtype == "quantized" else []
        run_sdk(guest_work, False, "inference", "--model", "model.json", "--model-data", "model.data",
                    "--device", "CPU", "--with-input-meta", "input.yml", "--postprocess-file", "post.yml",
                    "--dtype", dtype, "--batch-size", "1", "--iterations", "1",
                    "--output-dir", "sanity_" + dtype, *quantize)
    run_sdk(guest_work, True, "export", "ovxlib", "--model", "model.json", "--model-data", "model.data",
                "--dtype", "quantized", "--model-quantize", "model.quantize",
                "--target-ide-project", "linux64", "--with-input-meta", "input.yml",
                "--output-path", "wksp/point/point", "--save-fused-graph", "--postprocess-file", "post.yml",
                "--pack-nbg-unify", "--optimize", TARGET, "--viv-sdk", VIV_SDK)
    binary = work / "wksp/point_nbg_unify/network_binary.nb"
    if not binary.is_file() or binary.stat().st_size < 1024:
        raise RuntimeError("ACUITY did not produce a nonempty packed NB model")
    report.update(nb_size=binary.stat().st_size, nb_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for name in ("calibration_manifest.json", "input.yml", "post.yml", "model.quantize"):
        shutil.copy2(work / name, nb_path.parent / name)
    metadata = list(work.rglob("nbg_meta.json"))
    if len(metadata) != 1:
        raise RuntimeError("Expected exactly one exported NB metadata file")
    meta = json.loads(metadata[0].read_text())
    inputs, outputs = list(meta["Inputs"].values()), list(meta["Outputs"].values())
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Exported NB must have exactly one input/output")
    quant = inputs[0]["quantize"]
    if (inputs[0]["shape"] != [1, 3, 128, 128] or quant["qtype"] != "u8"
            or abs(quant["scale"] - 1 / 255) > 1e-8 or quant["zero_point"] != 0
            or outputs[0]["shape"] != [1, 11]):
        raise RuntimeError("Exported NB interface does not match the board")
    report["exported_metadata"] = meta
    shutil.copy2(metadata[0], nb_path.parent / "nbg_meta.json")
    (nb_path.parent / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary = nb_path.with_suffix(".nb.tmp")
    shutil.copy2(binary, temporary)
    temporary.replace(nb_path)
    print(f"NB ready: {nb_path} ({report['nb_size']} bytes), recipe={RECIPE}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: convert_nb.py INPUT.onnx DATASET OUTPUT.nb")
    main(*sys.argv[1:])
