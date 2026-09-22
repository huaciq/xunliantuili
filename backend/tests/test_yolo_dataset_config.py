from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.runs import _find_yolo_dataset_config


def test_find_yolo_dataset_config_supports_wrapping_directory(tmp_path: Path) -> None:
    config = tmp_path / "exported-dataset" / "data.yaml"
    config.parent.mkdir()
    config.write_text("train: images/train\n", encoding="utf-8")

    assert str(_find_yolo_dataset_config(tmp_path)) == (
        "/workspace/dataset/exported-dataset/data.yaml"
    )


def test_find_yolo_dataset_config_requires_exactly_one_file(tmp_path: Path) -> None:
    with pytest.raises(HTTPException, match="does not contain"):
        _find_yolo_dataset_config(tmp_path)

    first = tmp_path / "one" / "data.yaml"
    second = tmp_path / "two" / "data.yml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("train: images\n", encoding="utf-8")
    second.write_text("train: images\n", encoding="utf-8")

    with pytest.raises(HTTPException, match="multiple YOLO configuration files"):
        _find_yolo_dataset_config(tmp_path)
