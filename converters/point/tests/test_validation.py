import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("point_nb_validation", SOURCE / "nb_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def sdk_dumps(work):
    images = work / "validation128"
    images.mkdir()
    pixels = np.full((128, 128, 3), [32, 96, 224], dtype=np.uint8)
    Image.fromarray(pixels).save(images / "00000.png")
    prediction = np.array([.8, .9, .1, 0, 0, 0, 0, 0, 0, .5, .5])
    for stage in ("float32", "quantized"):
        folder = work / f"sanity_{stage}"
        folder.mkdir()
        np.savetxt(folder / "iter_0_input_out0_1_128_128_3.tensor", pixels.reshape(-1) / 255)
        np.savetxt(folder / "iter_0_output_out0_1_11.tensor", prediction)
    return prediction


@pytest.mark.parametrize("failure", [None, "nan", "missing", "float_error", "quantized_error"])
def test_inference_errors_prevent_accepting_a_conversion(tmp_path, failure):
    prediction = sdk_dumps(tmp_path)
    float_output = tmp_path / "sanity_float32/iter_0_output_out0_1_11.tensor"
    int8_output = tmp_path / "sanity_quantized/iter_0_output_out0_1_11.tensor"
    changed = prediction.copy()
    if failure == "nan":
        changed[0] = np.nan
        np.savetxt(int8_output, changed)
    if failure == "missing":
        int8_output.unlink()
    if failure == "float_error":
        changed[0] += .1
        np.savetxt(float_output, changed)
    if failure == "quantized_error":
        changed[-1] += .1
        np.savetxt(int8_output, changed)
    with patch.object(validation.ort, "InferenceSession") as session:
        session.return_value.get_inputs.return_value = [type("Port", (), {"name": "input"})()]
        session.return_value.run.return_value = [prediction[None]]
        call = lambda: validation.validate_predictions(
            tmp_path / "model.onnx", tmp_path,
            [{"source": "sample.jpg", "png": "00000.png"}], validation.DEFAULT_LIMITS,
        )
        if failure:
            with pytest.raises(ValueError):
                call()
        else:
            assert call()["status"] == "passed"
            report = json.loads((tmp_path / "quality_validation.json").read_text())
            assert report["sample_count"] == 1
            assert len(report["records"]) == 1


@pytest.mark.parametrize("limits", [None, {"max_probability_delta": -1}, {"unknown": 1}])
def test_invalid_quality_configuration_is_rejected(limits):
    with pytest.raises(ValueError):
        validation.quality_limits(limits)
