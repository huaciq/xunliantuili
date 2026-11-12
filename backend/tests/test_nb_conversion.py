import sys

import pytest
from fastapi.testclient import TestClient
from test_auth_projects import auth_header
from test_resources import _zip_bytes

from app.config import GLH_CONVERTER_COMMAND, settings
from app.main import app

# Avoid the Windows virtual environment launcher in the timeout test.
PYTHON = sys._base_executable


@pytest.fixture
def conversion_project(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("nb")
    storage = root / "storage"
    workdir = root / "glh converter"
    (workdir / "converter").mkdir(parents=True)
    (workdir / "sdk-marker").write_text("ready", encoding="utf-8")
    monkeypatch.setattr(settings, "storage_root", str(storage))
    monkeypatch.setattr(settings, "nb_conversion_projects", "振动盘转换验收")
    monkeypatch.setattr(settings, "nb_converter_workdir", str(workdir))
    monkeypatch.setattr(settings, "nb_converter_command", [PYTHON, "-c", "pass"])
    monkeypatch.setattr(settings, "nb_conversion_max_bytes", 128)
    with TestClient(app) as client:
        headers = auth_header(client)
        response = client.post(
            "/api/v1/projects", headers=headers, json={"name": "振动盘转换验收"}
        )
        assert response.status_code == 201
        assert response.json()["nb_conversion"] == "ready"
        project_path = f"/api/v1/projects/{response.json()['id']}"
        dataset = client.post(f"{project_path}/datasets", headers=headers, json={"name": "Point"})
        assert dataset.status_code == 201
        dataset_id = dataset.json()["id"]
        archive = _zip_bytes({
            "dataset200_point/train/images/001.jpg": "train-image",
            "dataset200_point/val/images/002.png": "validation-image",
            "dataset200_point/train/labels/001.txt": "0 0.5 0.5",
        })
        response = client.post(
            f"/api/v1/datasets/{dataset_id}/versions/upload",
            headers=headers, files={"file": ("point.zip", archive)},
        )
        assert response.status_code == 200, response.text
        version = response.json()["versions"][0]
        assert version["status"] == "ready"
        form = {"dataset_version_id": version["id"]}
        yield client, headers, f"{project_path}/conversions/onnx-to-nb", form
    assert not list((workdir / "converter" / "tasks").glob("platform-*"))


def test_download_contains_converter_output(conversion_project, monkeypatch):
    script = """\
from pathlib import Path
import json
import os
import sys

source, dataset, output = map(Path, sys.argv[1:])
assert source.read_bytes() == b"onnx-input"
assert Path("sdk-marker").read_text(encoding="utf-8") == "ready"
assert Path(os.environ["TMPDIR"]) == source.parent
assert source.parent.parent == Path.cwd() / "converter" / "tasks"
split = json.loads(source.with_name("training_split.json").read_text(encoding="utf-8"))
train, val = dataset / "train/images/001.jpg", dataset / "val/images/002.png"
assert split == {"train": [str(train)], "val": [str(val)]}
assert train.read_bytes() == b"train-image" and val.read_bytes() == b"validation-image"
assert not list(dataset.rglob("*.txt"))
assert output.name == "result.nb" and not output.exists()
output.write_bytes(b"nb-output")
"""
    monkeypatch.setattr(
        settings, "nb_converter_command",
        [PYTHON, "-c", script, "{input}", "{dataset}", "{output}"],
    )
    client, headers, path, form = conversion_project
    response = client.post(
        path, headers=headers, data=form, files={"file": ("点位.ONNX", b"onnx-input")}
    )

    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == (
        "attachment; filename*=utf-8''%E7%82%B9%E4%BD%8D.nb"
    )


@pytest.mark.parametrize(
    ("filename", "data", "status"),
    [("bad.pt", b"x", 422), ("empty.onnx", b"", 422), ("big.onnx", b"x" * 129, 413)],
)
def test_rejects_invalid_upload(conversion_project, filename, data, status):
    client, headers, path, form = conversion_project
    response = client.post(path, headers=headers, data=form, files={"file": (filename, data)})
    assert response.status_code == status


def test_needs_configured_converter_and_eligible_project(conversion_project, monkeypatch):
    client, headers, path, form = conversion_project
    files = {"file": ("model.onnx", b"onnx-input")}
    monkeypatch.setattr(settings, "nb_converter_command", list(GLH_CONVERTER_COMMAND))
    assert client.post(path, headers=headers, data=form, files=files).status_code == 503
    monkeypatch.setattr(settings, "nb_conversion_projects", "another-project")
    assert client.post(path, headers=headers, data=form, files=files).status_code == 404


def test_conversion_without_output_is_rejected(conversion_project):
    client, headers, path, form = conversion_project
    response = client.post(
        path, headers=headers, data=form, files={"file": ("model.onnx", b"onnx-input")}
    )
    assert response.status_code == 422
    assert response.headers.get("content-disposition") is None


def test_rejects_calibration_dataset_from_another_project(conversion_project, monkeypatch):
    client, headers, _, form = conversion_project
    project = client.post("/api/v1/projects", headers=headers, json={"name": "另一个振动盘项目"})
    assert project.status_code == 201
    monkeypatch.setattr(settings, "nb_conversion_projects", project.json()["id"])
    response = client.post(
        f"/api/v1/projects/{project.json()['id']}/conversions/onnx-to-nb",
        headers=headers, data=form, files={"file": ("model.onnx", b"onnx-input")},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "找不到该项目的数据集版本"


def test_timed_out_conversion_cleans_up_and_can_retry(conversion_project, monkeypatch):
    monkeypatch.setattr(settings, "nb_conversion_timeout_seconds", 1)
    monkeypatch.setattr(
        settings, "nb_converter_command", [PYTHON, "-c", "import time; time.sleep(30)"]
    )
    client, headers, path, form = conversion_project
    files = {"file": ("model.onnx", b"onnx-input")}
    response = client.post(path, headers=headers, data=form, files=files)
    assert response.status_code == 504, response.text
    monkeypatch.setattr(settings, "nb_converter_command", [PYTHON, "-c", "pass"])
    assert client.post(path, headers=headers, data=form, files=files).status_code == 422
