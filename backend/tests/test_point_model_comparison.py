import importlib.util
import math
from pathlib import Path

import pytest


def load_comparison():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_point_onnx.py"
    spec = importlib.util.spec_from_file_location("compare_point_onnx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output(confidence=0.9, category=0, x=0.5, y=0.5):
    probabilities = [0.0] * 8
    probabilities[category] = 1.0
    return [confidence, *probabilities, x, y]


def test_point_comparison_distinguishes_background_from_missing_label(tmp_path):
    module = load_comparison()
    label = tmp_path / "point.txt"
    with pytest.raises(FileNotFoundError):
        module.read_target(label)
    label.write_text("", encoding="utf-8")
    assert module.read_target(label) is None
    label.write_text("1 0.25 0.75", encoding="utf-8")
    assert module.read_target(label) == (1, 0.25, 0.75)
    for content in ("0.5 0.2 0.3", "8 0.2 0.3", "0 nan 0.3", "0 0.2 0.3 0.4 0.5"):
        label.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            module.read_target(label)


def test_point_comparison_checks_probability_and_finite_output_contract():
    module = load_comparison()
    assert module.validate_output(output())
    assert not module.validate_output([0.9, *([0.5] * 8), 0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        module.validate_output([math.nan, *output()[1:]])


def test_point_comparison_uses_padded_class_channels_and_final_xy():
    module = load_comparison()
    records = [
        {"reference": output(category=0), "candidate": output(category=1, x=0.75)},
        {"reference": output(0.1), "candidate": output()},
        {"reference": output(), "candidate": output(0.1, category=7)},
    ]
    summary = module.summarize(records, threshold=0.5, atol=1e-4, rtol=1e-3)
    assert not summary["all_outputs_close"]
    assert summary["presence_decisions_changed"] == 2
    assert summary["jointly_detected_images"] == 1
    assert summary["class_decisions_changed_on_joint_detections"] == 1
    assert summary["mean_point_shift_128px_on_joint_detections"] == 32


def test_point_comparison_reports_detection_and_label_metrics_separately():
    module = load_comparison()
    records = [
        {"target": (0, 0.5, 0.5), "candidate": output()},
        {"target": (1, 0.5, 0.5), "candidate": output(0.1, x=0.75)},
        {"target": None, "candidate": output()},
        {"target": None, "candidate": output(0.1)},
    ]
    metrics = module.task_metrics(records, "candidate", threshold=0.5)
    assert metrics["true_positives"] == metrics["false_negatives"] == 1
    assert metrics["false_positives"] == metrics["true_negatives"] == 1
    assert metrics["class_accuracy_on_all_positives"] == 0.5
    assert metrics["mean_point_error_128px_on_all_positives"] == 16
    assert metrics["presence_recall"] == metrics["background_false_positive_rate"] == 0.5


def test_point_comparison_tolerance_is_relative_to_reference():
    module = load_comparison()
    a, b = output(), output()
    b[9] += 0.0002
    assert module.summarize([{"reference": a, "candidate": b}], 0.5, 1e-4, 1e-3)[
        "all_outputs_close"
    ]
