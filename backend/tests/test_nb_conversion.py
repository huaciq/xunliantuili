import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_auth_projects import auth_header
from test_resources import _zip_bytes

from app.config import GLH_CONVERTER_COMMAND, Settings, settings
from app.database import SessionLocal
from app.main import app
from app.models import RunArtifact, TrainingRun, TrainingRunStatus
from app.services.results import collect_container_results

# Avoid the Windows virtual environment launcher in the timeout test.
PYTHON = sys._base_executable
FLAT_DATASET = _zip_bytes({
    "images/001.jpg": "train-image",
    "images/002.png": "validation-image",
    "images/003.jpg": "unused-image",
})


def _upload_dataset(client, headers, project_path, archive):
    dataset = client.post(f"{project_path}/datasets", headers=headers, json={"name": "Point"})
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]
    response = client.post(
        f"/api/v1/datasets/{dataset_id}/versions/upload",
        headers=headers, files={"file": ("point.zip", archive)},
    )
    assert response.status_code == 200, response.text
    version = response.json()["versions"][0]
    assert version["status"] == "ready"
    return version["id"]


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
        archive = _zip_bytes({
            "dataset200_point/train/images/001.jpg": "train-image",
            "dataset200_point/val/images/002.png": "validation-image",
            "dataset200_point/train/labels/001.txt": "0 0.5 0.5",
        })
        form = {"dataset_version_id": _upload_dataset(client, headers, project_path, archive)}
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


@pytest.fixture
def completed_training(conversion_project, monkeypatch):
    client, headers, _, _ = conversion_project
    project = client.post(
        "/api/v1/projects", headers=headers, json={"name": "振动盘点定位"}
    ).json()
    project_path = f"/api/v1/projects/{project['id']}"
    dataset_version_id = _upload_dataset(client, headers, project_path, FLAT_DATASET)
    code = client.post(f"{project_path}/code-packages", headers=headers, json={"name": "Trainer"})
    assert code.status_code == 201
    code_id = code.json()["id"]
    uploaded = client.post(
        f"/api/v1/code-packages/{code_id}/versions/upload", headers=headers,
        data={"default_entrypoint": "python train.py"},
        files={"file": ("code.zip", _zip_bytes({"train.py": "pass"}))},
    )
    assert uploaded.status_code == 200, uploaded.text
    templates = client.get("/api/v1/training-templates", headers=headers).json()
    template = next(item for item in templates if item["key"] == "custom_command")
    payload = {
        "name": "Point run",
        "template_id": template["id"],
        "dataset_version_id": dataset_version_id,
        "code_version_id": uploaded.json()["versions"][0]["id"],
        "parameters": {"command": ["python", "train.py"]},
    }
    created = client.post(f"{project_path}/runs", headers=headers, json=payload)
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]
    output_dir = Path(settings.storage_root) / "runs" / project["id"] / run_id / "output"
    output_dir.mkdir(parents=True)
    model = b"onnx-from-training"
    (output_dir / "point.onnx").write_bytes(model)
    (output_dir / "training_split.json").write_text(
        json.dumps({
            "train": ["/workspace/dataset/images/001.jpg"],
            "val": ["/workspace/dataset/images/002.png"],
        }),
        encoding="utf-8",
    )
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        run.status = TrainingRunStatus.SUCCEEDED
        collect_container_results(db, run)
        db.commit()

    script = """\
import json
import sys
from pathlib import Path

source, dataset, output = map(Path, sys.argv[1:])
split = json.loads(source.with_name("training_split.json").read_text(encoding="utf-8"))
assert source.read_bytes() == b"onnx-from-training"
assert [Path(p).read_bytes() for p in split["train"]] == [b"train-image"]
assert [Path(p).read_bytes() for p in split["val"]] == [b"validation-image"]
assert len(list(dataset.rglob("*.jpg"))) == 1
output.write_bytes(b"nb-output")
"""
    monkeypatch.setattr(
        settings, "nb_converter_command",
        [PYTHON, "-c", script, "{input}", "{dataset}", "{output}"],
    )
    artifacts = client.get(f"/api/v1/runs/{run_id}/artifacts", headers=headers).json()
    return {
        "project_path": project_path,
        "run_id": run_id,
        "onnx_id": next(item["id"] for item in artifacts if item["name"] == "point.onnx"),
        "split_id": next(item["id"] for item in artifacts if item["name"] == "training_split.json"),
        "model": model,
        "payload": payload,
    }


def test_reuses_actual_split_for_downloaded_onnx(
    conversion_project, completed_training, monkeypatch
):
    client, headers, path, _ = conversion_project
    model = completed_training["model"]
    response = client.post(
        path, headers=headers, files={"file": ("downloaded.onnx", model)}
    )
    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"
    assert response.headers["content-disposition"] == 'attachment; filename="downloaded.nb"'

    user_credentials = {"email": "calibrator@example.com", "password": "Calibrate123!"}
    user = client.post(
        "/api/v1/users", headers=headers, json={**user_credentials, "name": "Calibrator"},
    )
    assert user.status_code == 201
    target_project_path = path.removesuffix("/conversions/onnx-to-nb")
    member = client.post(
        f"{target_project_path}/members", headers=headers,
        json={"email": user_credentials["email"], "role": "researcher"},
    )
    assert member.status_code == 200
    login = client.post("/api/v1/auth/login", json=user_credentials)
    user_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    response = client.post(
        path, headers=user_headers, files={"file": ("downloaded.onnx", model)}
    )
    assert response.status_code == 422
    assert response.json()["detail"].startswith("找不到这份 ONNX 对应的训练划分")

    project_path = completed_training["project_path"]
    monkeypatch.setattr(settings, "nb_conversion_projects", project_path.rsplit("/", 1)[1])
    run_id, onnx_id = completed_training["run_id"], completed_training["onnx_id"]
    run_path = f"/api/v1/runs/{run_id}/artifacts/{onnx_id}/conversions/onnx-to-nb"
    assert client.post(run_path, headers=user_headers).status_code == 404

    for role, expected_status in (("viewer", 403), ("researcher", 200)):
        member = client.post(
            f"{project_path}/members", headers=headers,
            json={"email": user_credentials["email"], "role": role},
        )
        assert member.status_code == 200, member.text
        result = client.post(run_path, headers=user_headers)
        assert result.status_code == expected_status, result.text


def test_converts_run_artifact_with_its_original_split(
    conversion_project, completed_training, monkeypatch
):
    client, headers, _, _ = conversion_project
    run_id, onnx_id = completed_training["run_id"], completed_training["onnx_id"]
    path = f"/api/v1/runs/{run_id}/artifacts/{onnx_id}/conversions/onnx-to-nb"
    assert client.post(path, headers=headers).status_code == 404

    monkeypatch.setattr(
        settings, "nb_conversion_projects", Settings.model_fields["nb_conversion_projects"].default
    )
    response = client.post(path, headers=headers)
    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == 'attachment; filename="point.nb"'


def test_rejects_ineligible_run_artifacts(conversion_project, completed_training, monkeypatch):
    client, headers, _, _ = conversion_project
    monkeypatch.setattr(
        settings, "nb_conversion_projects", Settings.model_fields["nb_conversion_projects"].default
    )
    run_id, onnx_id = completed_training["run_id"], completed_training["onnx_id"]
    path = f"/api/v1/runs/{run_id}/artifacts/{onnx_id}/conversions/onnx-to-nb"
    unknown = f"/api/v1/runs/{run_id}/artifacts/unknown/conversions/onnx-to-nb"
    assert client.post(unknown, headers=headers).status_code == 404

    other = client.post(
        f"{completed_training['project_path']}/runs",
        headers=headers,
        json={**completed_training["payload"], "name": "Other run"},
    )
    assert other.status_code == 201, other.text
    other_id = other.json()["id"]
    with SessionLocal() as db:
        db.get(TrainingRun, other_id).status = TrainingRunStatus.SUCCEEDED
        db.commit()
    other_path = f"/api/v1/runs/{other_id}/artifacts/{onnx_id}/conversions/onnx-to-nb"
    assert client.post(other_path, headers=headers).status_code == 404

    with SessionLocal() as db:
        db.get(TrainingRun, run_id).status = TrainingRunStatus.FAILED
        db.commit()
    assert client.post(path, headers=headers).status_code == 409

    with SessionLocal() as db:
        db.get(TrainingRun, run_id).status = TrainingRunStatus.SUCCEEDED
        onnx = db.get(RunArtifact, onnx_id)
        onnx.details = {**onnx.details, "development_artifact": True}
        db.commit()
    assert client.post(path, headers=headers).status_code == 422

    with SessionLocal() as db:
        onnx = db.get(RunArtifact, onnx_id)
        onnx.details = {**onnx.details, "development_artifact": False}
        db.delete(db.get(RunArtifact, completed_training["split_id"]))
        db.commit()
    assert client.post(path, headers=headers).status_code == 422


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
