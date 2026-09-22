import pytest

from app.services.scheduler import _yolo_workdir


def test_yolo_workdir_uses_dataset_config_parent() -> None:
    assert _yolo_workdir(["yolo", "data=/workspace/dataset/data.yaml"]) == (
        "/workspace/dataset"
    )
    assert _yolo_workdir(
        ["yolo", "data=/workspace/dataset/exported-dataset/data.yaml"]
    ) == "/workspace/dataset/exported-dataset"


@pytest.mark.parametrize(
    "command",
    [
        ["yolo"],
        ["yolo", "data=/tmp/data.yaml"],
        ["yolo", "data=relative/data.yaml"],
        ["yolo", "data=/workspace/dataset/a.yaml", "data=/workspace/dataset/b.yaml"],
    ],
)
def test_yolo_workdir_rejects_missing_or_unsafe_data_argument(command: list[str]) -> None:
    with pytest.raises(ValueError):
        _yolo_workdir(command)
