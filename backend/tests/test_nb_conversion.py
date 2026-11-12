import sys

import pytest
from fastapi.testclient import TestClient
from test_auth_projects import auth_header

from app.config import settings
from app.main import app


@pytest.fixture
def conversion_project(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    workdir = tmp_path / "glh converter"
    workdir.mkdir()
    (workdir / "sdk-marker").write_text("ready", encoding="utf-8")
    monkeypatch.setattr(settings, "storage_root", str(storage))
    monkeypatch.setattr(settings, "nb_conversion_projects", "振动盘转换验收")
    monkeypatch.setattr(settings, "nb_converter_workdir", str(workdir))
    monkeypatch.setattr(settings, "nb_converter_command", [sys.executable, "-c", "pass"])
    monkeypatch.setattr(settings, "nb_conversion_max_bytes", 128)
    with TestClient(app) as client:
        headers = auth_header(client)
        response = client.post(
            "/api/v1/projects", headers=headers, json={"name": "振动盘转换验收"}
        )
        assert response.status_code == 201
        assert response.json()["nb_conversion"] == "ready"
        path = f"/api/v1/projects/{response.json()['id']}/conversions/onnx-to-nb"
        yield client, headers, path
    assert not storage.exists() or list(storage.iterdir()) == []


def test_download_contains_converter_output(conversion_project, monkeypatch):
    script = """\
from pathlib import Path
import os
import sys

source, output = map(Path, sys.argv[1:])
assert source.read_bytes() == b"onnx-input"
assert Path("sdk-marker").read_text(encoding="utf-8") == "ready"
assert Path(os.environ["TMPDIR"]) == source.parent
output.write_bytes(b"nb-output")
"""
    monkeypatch.setattr(
        settings, "nb_converter_command", [sys.executable, "-c", script, "{input}", "{output}"]
    )
    client, headers, path = conversion_project
    response = client.post(path, headers=headers, files={"file": ("点位.ONNX", b"onnx-input")})

    assert response.status_code == 200
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
    client, headers, path = conversion_project
    response = client.post(path, headers=headers, files={"file": (filename, data)})
    assert response.status_code == status


def test_needs_configured_converter_and_eligible_project(conversion_project, monkeypatch):
    client, headers, path = conversion_project
    files = {"file": ("model.onnx", b"onnx-input")}
    monkeypatch.setattr(settings, "nb_converter_command", [])
    assert client.post(path, headers=headers, files=files).status_code == 503
    monkeypatch.setattr(settings, "nb_conversion_projects", "another-project")
    assert client.post(path, headers=headers, files=files).status_code == 404


def test_conversion_without_output_is_rejected(conversion_project):
    client, headers, path = conversion_project
    response = client.post(path, headers=headers, files={"file": ("model.onnx", b"onnx-input")})
    assert response.status_code == 422
    assert response.headers.get("content-disposition") is None


def test_timed_out_conversion_cleans_up_and_can_retry(conversion_project, monkeypatch):
    monkeypatch.setattr(settings, "nb_conversion_timeout_seconds", 1)
    monkeypatch.setattr(
        settings, "nb_converter_command", [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    client, headers, path = conversion_project
    files = {"file": ("model.onnx", b"onnx-input")}
    assert client.post(path, headers=headers, files=files).status_code == 504
    monkeypatch.setattr(settings, "nb_converter_command", [sys.executable, "-c", "pass"])
    assert client.post(path, headers=headers, files=files).status_code == 422
