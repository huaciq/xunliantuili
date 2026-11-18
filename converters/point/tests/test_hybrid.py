import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import convert_nb as converter
import hybrid_quantization as hybrid


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def baseline(tmp_path):
    graph = {"Layers": {"input": {"op": "input"}, "head": {"op": "sigmoid"},
                        "output": {"op": "output"}}}
    write_json(tmp_path / "model.json", graph)
    (tmp_path / "model.data").write_bytes(b"baseline weights")
    table = {"customized_quantize_layers": {"head": "dynamic_fixed_point-i16"},
             "quantize_parameters": {"@input:out0": {"qtype": "i8"}}}
    write_json(tmp_path / "model.quantize.before_input_u8", table)
    table["quantize_parameters"]["@input:out0"]["qtype"] = "u8"
    write_json(tmp_path / "model.quantize", table)
    for name in ("input.yml", "validation_input.yml", "post.yml"):
        (tmp_path / name).write_text("original metadata")
    for name in ("sanity_float32", "sanity_quantized"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "prediction.tensor").write_text("baseline prediction")
    write_json(tmp_path / "quality_validation.json", {
        "status": "failed", "records": [{"float_passed": True, "int8_passed": False}],
    })
    return tmp_path


@pytest.mark.parametrize("failure", [None, "float", "no_recommendations", "incomplete"])
def test_retry_requires_valid_float_results_and_sdk_recommendations(baseline, failure):
    if failure == "float":
        write_json(baseline / "quality_validation.json", {
            "records": [{"float_passed": False, "int8_passed": False}],
        })
    if failure == "no_recommendations":
        write_json(baseline / "model.quantize", {"customized_quantize_layers": {}})
    if failure == "incomplete":
        (baseline / "quality_validation.json").unlink()
    assert hybrid.can_retry(baseline) is (failure is None)


@pytest.mark.parametrize("graphs", [0, 1, 2])
def test_hybrid_preserves_baseline_and_selects_new_graph_and_data(baseline, graphs):
    old_graph = (baseline / "model.json").read_bytes()

    def sdk_runner(guest, exporting, *args):
        assert not exporting and args[0] == "quantize"
        assert "--hybrid" in args and "--rebuild" not in args
        assert args[args.index("--iterations") + 1] == "160"
        table = yaml.safe_load((baseline / "model.quantize").read_text())
        assert table["quantize_parameters"]["@input:out0"]["qtype"] == "i8"
        (baseline / "model.data").write_bytes(b"hybrid weights")
        table["quantize_parameters"]["@head:out0"] = {"qtype": "i16"}
        write_json(baseline / "model.quantize", table)
        for i in range(graphs):
            write_json(baseline / f"hybrid_{i}.json", {"Layers": {"hybrid_head": {"op": "sigmoid"}}})

    if graphs != 1:
        with pytest.raises(ValueError, match="Expected one new SDK hybrid model graph"):
            hybrid.apply(baseline, Path("/workspace/job"), sdk_runner, 160)
    else:
        result = hybrid.apply(baseline, Path("/workspace/job"), sdk_runner, 160)
        assert result["sdk_graph"] == "hybrid_0.json"
        assert "hybrid_head" in json.loads((baseline / "model.json").read_text())["Layers"]
        assert (baseline / "model.data").read_bytes() == b"hybrid weights"
        assert not (baseline / "quality_validation.json").exists()
        assert not (baseline / "sanity_quantized").exists()
    backup = baseline / "baseline_int8"
    assert (backup / "model.json").read_bytes() == old_graph
    assert (backup / "model.data").read_bytes() == b"baseline weights"
    assert (backup / "quality_validation.json").is_file()
    assert (backup / "sanity_quantized/prediction.tensor").read_text() == "baseline prediction"


@pytest.mark.parametrize("second_passes", [True, False])
def test_export_requires_hybrid_revalidation(tmp_path, monkeypatch, second_passes):
    root = tmp_path / "converter"
    root.mkdir()
    model = root / "input.onnx"
    model.write_bytes(b"source")
    (root / "sdk_point_entry.py").write_text("# mock SDK entry")
    target = root / "result.nb"
    monkeypatch.setattr(converter, "ROOT", root)
    monkeypatch.setattr(converter, "split_images", lambda *args: ([model], [model]))
    monkeypatch.setattr(converter, "clean_head", lambda *args: {})
    monkeypatch.setattr(converter, "verify_equivalence", lambda *args: {})

    def images(files, work, guest, **kwargs):
        for name in ("calibration_manifest.json", "validation_manifest.json"):
            write_json(work / name, [])
        return []

    monkeypatch.setattr(converter, "prepare_images", images)
    monkeypatch.setattr(converter.onnx, "load", lambda *args: SimpleNamespace(graph=SimpleNamespace(
        input=[SimpleNamespace(name="input")], output=[SimpleNamespace(name="output")],
    )))
    events = []

    def run_sdk(guest, exporting, *args):
        work = tmp_path / "sdk" / guest.relative_to("/")
        if args[0] == "import":
            write_json(work / "model.json", {"Layers": {
                "input": {"op": "input"}, "head": {"op": "sigmoid"}, "output": {"op": "output"},
            }})
            (work / "model.data").write_bytes(b"baseline weights")
        if args[0] == "quantize":
            if "--hybrid" in args:
                events.append("hybrid")
                graph = json.loads((work / "model.json").read_text())
                graph["Layers"]["hybrid_head"] = {"op": "sigmoid"}
                write_json(work / "quantize.json", graph)
                (work / "model.data").write_bytes(b"hybrid weights")
            write_json(work / "model.quantize", {
                "customized_quantize_layers": {"head": "dynamic_fixed_point-i16"},
                "quantize_parameters": {"@input:out0": {"qtype": "i8"}},
            })
        if args[0] == "inference":
            is_hybrid = (work / "baseline_int8").exists()
            dtype = args[args.index("--dtype") + 1]
            events.append(("hybrid" if is_hybrid else "baseline") + "_" + dtype)
            if is_hybrid:
                assert "hybrid_head" in json.loads((work / "model.json").read_text())["Layers"]
                assert (work / "model.data").read_bytes() == b"hybrid weights"
            (work / ("sanity_" + dtype)).mkdir()
        if args[0] == "export":
            events.append("export")
            assert second_passes
            assert (work / "model.data").read_bytes() == b"hybrid weights"
            output = work / "wksp/point_nbg_unify"
            output.mkdir(parents=True)
            (output / "network_binary.nb").write_bytes(b"n" * 2048)
            write_json(output / "nbg_meta.json", {})

    def validate(model, work, records, limits):
        passed = (work / "baseline_int8").exists() and second_passes
        report = {"status": "passed" if passed else "failed", "records": [
            {"float_passed": True, "int8_passed": passed},
        ]}
        write_json(work / "quality_validation.json", report)
        if not passed:
            raise ValueError("SDK INT8 quantization exceeds error limits")
        return report

    monkeypatch.setattr(converter, "validate_predictions", validate)
    monkeypatch.setattr(converter, "validate_nb_interface", lambda *args: {"output_dtype": "float16"})
    if second_passes:
        converter.main(model, root, target, workspace_root=tmp_path / "sdk", sdk_runner=run_sdk)
        assert target.is_file()
        report = json.loads((root / "conversion_report.json").read_text())
        assert report["quantization_strategy"] == "sdk_recommended_hybrid"
    else:
        with pytest.raises(RuntimeError, match="NB quality validation failed"):
            converter.main(model, root, target, workspace_root=tmp_path / "sdk", sdk_runner=run_sdk)
        assert not target.exists()
        assert "export" not in events
    assert events[:5] == ["baseline_float32", "baseline_quantized", "hybrid",
                          "hybrid_float32", "hybrid_quantized"]
