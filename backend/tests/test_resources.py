import io
import json
import uuid
import zipfile

from fastapi.testclient import TestClient

from app.main import app


def _login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _zip_bytes(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_dataset_and_code_version_uploads() -> None:
    with TestClient(app) as client:
        headers = _login(client)
        suffix = uuid.uuid4().hex[:8]
        project = client.post(
            "/api/v1/projects",
            headers=headers,
            json={"name": f"Vision {suffix}", "description": "Resource upload tests"},
        ).json()

        dataset_response = client.post(
            f"/api/v1/projects/{project['id']}/datasets",
            headers=headers,
            json={"name": "Surface defects", "description": "YOLO dataset"},
        )
        assert dataset_response.status_code == 201
        dataset = dataset_response.json()

        dataset_archive = _zip_bytes(
            {
                "data.yaml": "path: .\ntrain: images/train\nval: images/val\n",
                "images/train/sample.jpg": "fake-image",
                "labels/train/sample.txt": "0 0.5 0.5 0.2 0.2",
            }
        )
        upload_response = client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/upload",
            headers=headers,
            files={"file": ("dataset.zip", dataset_archive, "application/zip")},
        )
        assert upload_response.status_code == 200
        version = upload_response.json()["versions"][0]
        assert version["status"] == "ready"
        assert version["format"] == "yolo"
        assert version["file_count"] == 3

        manifest_response = client.get(
            f"/api/v1/dataset-versions/{version['id']}/manifest", headers=headers
        )
        assert manifest_response.status_code == 200
        manifest = manifest_response.json()
        assert manifest["dataset_format"] == "yolo"
        assert {entry["path"] for entry in manifest["files"]} >= {
            "data.yaml",
            "images/train/sample.jpg",
        }

        code_response = client.post(
            f"/api/v1/projects/{project['id']}/code-packages",
            headers=headers,
            json={"name": "YOLO trainer", "description": "Training source"},
        )
        assert code_response.status_code == 201
        code_package = code_response.json()
        code_archive = _zip_bytes({"train.py": "print('train')", "config.json": json.dumps({})})
        code_upload = client.post(
            f"/api/v1/code-packages/{code_package['id']}/versions/upload",
            headers=headers,
            data={"default_workdir": ".", "default_entrypoint": "python train.py"},
            files={"file": ("source.zip", code_archive, "application/zip")},
        )
        assert code_upload.status_code == 200
        code_version = code_upload.json()["versions"][0]
        assert code_version["status"] == "ready"
        assert code_version["default_entrypoint"] == "python train.py"


def test_archive_path_traversal_is_rejected() -> None:
    with TestClient(app) as client:
        headers = _login(client)
        suffix = uuid.uuid4().hex[:8]
        project = client.post(
            "/api/v1/projects",
            headers=headers,
            json={"name": f"Unsafe {suffix}", "description": "Archive security test"},
        ).json()
        dataset = client.post(
            f"/api/v1/projects/{project['id']}/datasets",
            headers=headers,
            json={"name": "Unsafe archive", "description": "Must fail"},
        ).json()
        malicious = _zip_bytes({"../outside.txt": "should-not-extract"})
        response = client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/upload",
            headers=headers,
            files={"file": ("unsafe.zip", malicious, "application/zip")},
        )
        assert response.status_code == 400
        assert "Unsafe archive path" in response.json()["detail"]
