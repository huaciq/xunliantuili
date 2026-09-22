import io
import time
import uuid
import zipfile

from fastapi.testclient import TestClient

from app.main import app


def _zip(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _ready_resources(
    client: TestClient, headers: dict[str, str], project_id: str
) -> tuple[str, str]:
    dataset = client.post(
        f"/api/v1/projects/{project_id}/datasets",
        headers=headers,
        json={"name": "Training data", "description": "YOLO"},
    ).json()
    dataset_result = client.post(
        f"/api/v1/datasets/{dataset['id']}/versions/upload",
        headers=headers,
        files={
            "file": (
                "dataset.zip",
                _zip(
                    {
                        "wrapped-dataset/data.yaml": "train: images",
                        "wrapped-dataset/images/a.jpg": "image",
                    }
                ),
                "application/zip",
            )
        },
    ).json()
    code = client.post(
        f"/api/v1/projects/{project_id}/code-packages",
        headers=headers,
        json={"name": "Trainer", "description": "Code"},
    ).json()
    code_result = client.post(
        f"/api/v1/code-packages/{code['id']}/versions/upload",
        headers=headers,
        data={"default_entrypoint": "python train.py"},
        files={"file": ("code.zip", _zip({"train.py": "print('ok')"}), "application/zip")},
    ).json()
    return dataset_result["versions"][0]["id"], code_result["versions"][0]["id"]


def test_training_run_lifecycle_and_gpu_release() -> None:
    with TestClient(app) as client:
        headers = _headers(client)
        suffix = uuid.uuid4().hex[:8]
        project = client.post(
            "/api/v1/projects",
            headers=headers,
            json={"name": f"Training {suffix}", "description": "Scheduler test"},
        ).json()
        dataset_version_id, code_version_id = _ready_resources(client, headers, project["id"])
        templates = client.get("/api/v1/training-templates", headers=headers).json()
        yolo = next(item for item in templates if item["key"] == "yolo_detection")
        response = client.post(
            f"/api/v1/projects/{project['id']}/runs",
            headers=headers,
            json={
                "name": "YOLO baseline",
                "template_id": yolo["id"],
                "dataset_version_id": dataset_version_id,
                "code_version_id": code_version_id,
                "parameters": {"epochs": 5, "batch": 4, "imgsz": 640, "model": "yolo11n.pt"},
                "requested_gpu_count": 2,
                "requested_gpu_model": "rtx_4090",
            },
        )
        assert response.status_code == 201
        run = response.json()
        assert run["status"] == "draft"
        assert run["command"][:3] == ["yolo", "detect", "train"]
        assert "data=/workspace/dataset/wrapped-dataset/data.yaml" in run["command"]
        assert "model=/opt/models/yolo11n.pt" in run["command"]

        submitted = client.post(f"/api/v1/runs/{run['id']}/submit", headers=headers)
        assert submitted.status_code == 200
        assert submitted.json()["status"] == "queued"

        deadline = time.monotonic() + 4
        final = None
        saw_running = False
        while time.monotonic() < deadline:
            final = client.get(f"/api/v1/runs/{run['id']}", headers=headers).json()
            if final["status"] == "running":
                saw_running = True
                assert len(final["allocated_gpus"]) == 2
            if final["status"] == "succeeded":
                break
            time.sleep(0.08)
        assert saw_running
        assert final is not None and final["status"] == "succeeded"
        assert final["exit_code"] == 0

        logs = client.get(f"/api/v1/runs/{run['id']}/logs", headers=headers).json()
        assert any("Training completed successfully" in line for line in logs["lines"])
        gpus = client.get("/api/v1/resources/gpus", headers=headers).json()
        assert all(gpu["allocated_run_id"] is None for gpu in gpus)

        metrics = client.get(f"/api/v1/runs/{run['id']}/metrics", headers=headers).json()
        assert len(metrics) == 20
        assert {point["key"] for point in metrics} == {
            "train/loss",
            "val/map50",
            "val/precision",
            "val/recall",
        }
        artifacts = client.get(f"/api/v1/runs/{run['id']}/artifacts", headers=headers).json()
        checkpoint = next(item for item in artifacts if item["artifact_type"] == "checkpoint")
        checkpoint_download = client.get(
            f"/api/v1/artifacts/{checkpoint['id']}/download", headers=headers
        )
        assert checkpoint_download.status_code == 200
        assert b'"development_artifact": true' in checkpoint_download.content

        registered = client.post(
            f"/api/v1/runs/{run['id']}/register-model",
            headers=headers,
            json={"name": "Defect detector", "description": "M4 registry test"},
        )
        assert registered.status_code == 201
        model = registered.json()
        assert model["versions"][0]["version"] == 1
        assert model["versions"][0]["metrics"]["val/map50"] == 0.92

        exported = client.post(
            f"/api/v1/model-versions/{model['versions'][0]['id']}/exports",
            headers=headers,
            json={"format": "onnx"},
        )
        assert exported.status_code == 201
        export = exported.json()
        assert export["status"] == "succeeded"
        assert export["validation_status"] == "passed"
        assert export["max_abs_diff"] < 0.001
        export_download = client.get(
            f"/api/v1/artifacts/{export['artifact_id']}/download", headers=headers
        )
        assert export_download.content.startswith(b"FAKE_ONNX")

        models = client.get(
            f"/api/v1/projects/{project['id']}/models", headers=headers
        ).json()
        assert models[0]["name"] == "Defect detector"
        assert models[0]["versions"][0]["exports"][0]["status"] == "succeeded"

        stop_payload = {
            "name": "YOLO stoppable",
            "template_id": yolo["id"],
            "dataset_version_id": dataset_version_id,
            "code_version_id": code_version_id,
            "parameters": {"epochs": 5, "batch": 4, "imgsz": 640, "model": "yolo11n.pt"},
            "requested_gpu_count": 1,
            "requested_gpu_model": "rtx_3090",
        }
        stoppable = client.post(
            f"/api/v1/projects/{project['id']}/runs",
            headers=headers,
            json=stop_payload,
        ).json()
        client.post(f"/api/v1/runs/{stoppable['id']}/submit", headers=headers)

        deadline = time.monotonic() + 2
        running = None
        while time.monotonic() < deadline:
            running = client.get(f"/api/v1/runs/{stoppable['id']}", headers=headers).json()
            if running["status"] == "running":
                break
            time.sleep(0.03)
        assert running is not None and running["status"] == "running"

        stopped = client.post(f"/api/v1/runs/{stoppable['id']}/stop", headers=headers)
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "cancel_requested"

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stopped_run = client.get(
                f"/api/v1/runs/{stoppable['id']}", headers=headers
            ).json()
            if stopped_run["status"] == "stopped":
                break
            time.sleep(0.03)
        assert stopped_run["status"] == "stopped"
        assert stopped_run["exit_code"] == 143
        stop_logs = client.get(
            f"/api/v1/runs/{stoppable['id']}/logs", headers=headers
        ).json()
        assert any("Execution stopped" in line for line in stop_logs["lines"])
        gpus = client.get("/api/v1/resources/gpus", headers=headers).json()
        assert all(gpu["allocated_run_id"] is None for gpu in gpus)
