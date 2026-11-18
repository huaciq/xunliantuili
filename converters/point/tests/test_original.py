import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import convert_nb as converter


@pytest.mark.parametrize("entry", json.loads((SOURCE / "original_source.json").read_text())["files"])
def test_original_sources_only_have_documented_docker_permission_adaptations(entry):
    data = (SOURCE / entry["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["platform_sha256"]
    if entry["file"] == "convert_nb_docker.py":
        old, new = b'["sudo", "-n", "docker", "run"', b'["docker", "run"'
        assert data.count(new) == 1
        data = data.replace(new, old)
    elif entry["file"] == "sdk_container_watchdog.py":
        old, new = b"['sudo', '-n', 'docker', 'rm'", b"['docker', 'rm'"
        assert data.count(new) == 1
        data = data.replace(new, old)
    assert hashlib.sha256(data).hexdigest() == entry["original_sha256"]


@pytest.mark.parametrize("failure", [None, "sdk", "missing_binary", "wrong_interface"])
def test_original_sdk_flow_and_publish_conditions(tmp_path, monkeypatch, failure):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"source ONNX")
    (tmp_path / "sdk_point_entry.py").write_text("# mock")
    output = tmp_path / "result.nb"
    monkeypatch.setattr(converter, "ROOT", tmp_path)
    monkeypatch.setattr(converter, "calibration_images", lambda *args: [model, model])
    monkeypatch.setattr(converter, "clean_head", lambda *args: {})
    monkeypatch.setattr(converter, "verify_equivalence", lambda *args: {})

    def prepare(files, work, guest):
        (work / "calibration_manifest.json").write_text("[]")

    monkeypatch.setattr(converter, "prepare_images", prepare)
    monkeypatch.setattr(converter.onnx, "load", lambda *args: SimpleNamespace(graph=SimpleNamespace(
        input=[SimpleNamespace(name="input")], output=[SimpleNamespace(name="output")],
    )))
    stages = []

    def sdk(guest, exporting, *args):
        work = tmp_path / "sdk" / guest.relative_to("/")
        stages.append(args[0])
        if args[0] == "import":
            (work / "model.json").write_text(json.dumps({"Layers": {
                "input": {"op": "input"}, "output": {"op": "output"},
            }}))
        if args[0] == "quantize":
            assert "--rebuild" in args and "--hybrid" not in args
            assert args[args.index("--iterations") + 1] == "2"
            assert args[args.index("--quantizer") + 1] == "perchannel_symmetric_affine"
            assert args[args.index("--qtype") + 1] == "int8"
            (work / "model.quantize").write_text(json.dumps({
                "quantize_parameters": {"@input:out0": {"qtype": "i8"}},
            }))
        if args[0] == "inference":
            assert args[args.index("--iterations") + 1] == "1"
            assert args[args.index("--with-input-meta") + 1] == "input.yml"
            post = yaml.safe_load((work / "post.yml").read_text())
            assert post["postprocess"]["app_postprocs"][0]["postproc_params"]["force_float32"] is True
            if failure == "sdk":
                raise subprocess.CalledProcessError(1, args)
        if args[0] == "export":
            assert exporting
            assert args[args.index("--optimize") + 1] == "VIP9000NANODI_PLUS_PID0X1000003B"
            exported = work / "wksp/point_nbg_unify"
            exported.mkdir(parents=True)
            if failure != "missing_binary":
                (exported / "network_binary.nb").write_bytes(b"n" * 2048)
            meta = {"Inputs": {"input": {"shape": [1, 3, 128, 128], "quantize": {
                "qtype": "u8", "scale": .1 if failure == "wrong_interface" else 1 / 255,
                "zero_point": 0,
            }}}, "Outputs": {"output": {"shape": [1, 11]}}}
            (exported / "nbg_meta.json").write_text(json.dumps(meta))

    if failure:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            converter.main(model, tmp_path, output, workspace_root=tmp_path / "sdk", sdk_runner=sdk)
        assert not output.exists()
    else:
        converter.main(model, tmp_path, output, workspace_root=tmp_path / "sdk", sdk_runner=sdk)
        assert stages == ["import", "quantize", "inference", "inference", "export"]
        assert output.read_bytes() == b"n" * 2048
        report = json.loads((tmp_path / "conversion_report.json").read_text())
        assert report["recipe"] == "point-pcint8-rgb-u8-v1"
