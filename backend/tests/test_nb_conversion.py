import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_auth_projects import auth_header

from app.config import settings
from app.main import app
from app.services import nb_conversion


@pytest.fixture
def conversion_project(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))
    monkeypatch.setattr(settings, "nb_conversion_projects", "振动盘转换验收")
    monkeypatch.setattr(settings, "nb_converter_image", "test/nb-converter:local")
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
    assert list(tmp_path.iterdir()) == []


def test_download_contains_converter_output(conversion_project, monkeypatch):
    def run(argv, **kwargs):
        assert "--network=none" in argv
        assert argv[-3:] == [settings.nb_converter_image, "/input/model.onnx", "/output/model.nb"]
        mounts = [arg.split(",")[1:] for arg in argv if arg.startswith("type=bind,")]
        assert mounts[0][1:] == ["target=/input/model.onnx", "readonly"]
        source = Path(mounts[0][0].removeprefix("source="))
        assert source.read_bytes() == b"onnx-input"
        output = Path(mounts[1][0].removeprefix("source=")) / "model.nb"
        output.write_bytes(b"nb-output")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(nb_conversion.subprocess, "run", run)
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
    monkeypatch.setattr(settings, "nb_converter_image", "")
    assert client.post(path, headers=headers, files=files).status_code == 503
    monkeypatch.setattr(settings, "nb_conversion_projects", "another-project")
    assert client.post(path, headers=headers, files=files).status_code == 404


def test_conversion_without_output_is_rejected(conversion_project, monkeypatch):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(nb_conversion.subprocess, "run", run)
    client, headers, path = conversion_project
    response = client.post(path, headers=headers, files={"file": ("model.onnx", b"onnx-input")})
    assert response.status_code == 422
    assert response.headers.get("content-disposition") is None


def test_timed_out_conversion_removes_container(conversion_project, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(nb_conversion.subprocess, "run", run)
    client, headers, path = conversion_project
    response = client.post(path, headers=headers, files={"file": ("model.onnx", b"onnx-input")})
    assert response.status_code == 504
    assert calls[1][1:] == ["rm", "-f", calls[0][calls[0].index("--name") + 1]]
