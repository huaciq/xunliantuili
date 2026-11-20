"""Exercise progress while the real converter subprocess is still running."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from test_nb_conversion import PYTHON

from app.config import settings
from app.services.nb_conversion_progress import ConversionLogProgress, start_progress

pytest_plugins = ("test_nb_conversion",)


def progress_path(conversion_path, conversion_id):
    return f"{conversion_path}/progress/{conversion_id}"


def tracked_headers(headers, conversion_id):
    return {**headers, "X-NB-Conversion-ID": str(conversion_id)}


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.025)
    raise AssertionError("Expected progress was not observable before the converter finished")


def test_progress_is_live_monotonic_and_only_finishes_after_nb_exists(
    conversion_project, monkeypatch
):
    client, headers, path, form = conversion_project
    workdir = Path(settings.nb_converter_workdir)
    quantize_ready = workdir / "quantize.ready"
    release_quantize = workdir / "quantize.release"
    export_ready = workdir / "export.ready"
    release_export = workdir / "export.release"
    script = """\
import sys
import time
from pathlib import Path

(source, dataset, output, quantize_ready, release_quantize,
 export_ready, release_export) = map(Path, sys.argv[1:])
def wait_for(path):
    deadline = time.monotonic() + 15
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Test did not release converter gate")
        time.sleep(.025)
print("SDK native stage: import", flush=True)
print("SDK native stage: quantize", flush=True)
quantize_ready.write_text("ready")
wait_for(release_quantize)
print("SDK native stage: inference", flush=True)
print("SDK native stage: inference", flush=True)
print("SDK native stage: export", flush=True)
export_ready.write_text("ready")
wait_for(release_export)
output.write_bytes(b"nb-output")
"""
    monkeypatch.setattr(settings, "nb_conversion_timeout_seconds", 30)
    monkeypatch.setattr(
        settings,
        "nb_converter_command",
        [
            PYTHON,
            "-c",
            script,
            "{input}",
            "{dataset}",
            "{output}",
            *map(str, (quantize_ready, release_quantize, export_ready, release_export)),
        ],
    )
    conversion_id = uuid4()
    url = progress_path(path, conversion_id)

    def at_stage(stage):
        response = client.get(url, headers=headers)
        if response.status_code == 404:
            return None
        assert response.status_code == 200, response.text
        body = response.json()
        return body if body["stage"] == stage else None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.post,
            path,
            headers=tracked_headers(headers, conversion_id),
            data=form,
            files={"file": ("point.onnx", b"onnx-input")},
        )
        try:
            wait_until(quantize_ready.exists)
            quantize = wait_until(lambda: at_stage("quantize"))
            assert quantize["status"] == "running"
            assert 0 < quantize["percent"] < 100
            assert quantize["message"] and quantize["updated_at"]
            assert not future.done()
            release_quantize.write_text("continue")
            wait_until(export_ready.exists)
            exporting = wait_until(lambda: at_stage("export"))
            assert exporting["status"] == "running"
            assert quantize["percent"] < exporting["percent"] < 100
            assert not future.done()
        finally:
            release_quantize.write_text("continue")
            release_export.write_text("continue")
        response = future.result(timeout=10)
    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"
    finished = client.get(url, headers=headers)
    assert finished.status_code == 200, finished.text
    assert finished.headers["cache-control"] == "no-store"
    assert finished.json()["status"] == "succeeded"
    assert finished.json()["percent"] == 100
    # The existing temporary model workspace has already been removed, but the
    # final progress record must remain readable by the browser.
    assert not list((workdir / "converter/tasks").glob("platform-*"))


@pytest.mark.parametrize("failure", ["sdk_error", "timeout", "missing_output"])
def test_failed_or_timed_out_conversion_never_reports_100(conversion_project, monkeypatch, failure):
    client, headers, path, form = conversion_project
    scripts = {
        "sdk_error": "import sys; print('SDK native stage: export', flush=True); sys.exit(1)",
        "timeout": "import time; print('SDK native stage: quantize', flush=True); time.sleep(30)",
        "missing_output": "print('SDK native stage: export', flush=True)",
    }
    monkeypatch.setattr(
        settings, "nb_conversion_timeout_seconds", 1 if failure == "timeout" else 10
    )
    monkeypatch.setattr(settings, "nb_converter_command", [PYTHON, "-c", scripts[failure]])
    conversion_id = uuid4()
    response = client.post(
        path,
        headers=tracked_headers(headers, conversion_id),
        data=form,
        files={"file": ("point.onnx", b"onnx-input")},
    )
    assert response.status_code == (504 if failure == "timeout" else 422), response.text
    progress = client.get(progress_path(path, conversion_id), headers=headers)
    assert progress.status_code == 200, progress.text
    assert progress.json()["status"] == "failed"
    assert progress.json()["percent"] < 100
    assert progress.json()["message"]
    assert not list((Path(settings.nb_converter_workdir) / "converter/tasks").glob("platform-*"))


def test_progress_is_private_and_duplicate_id_cannot_overwrite_success(
    conversion_project, monkeypatch
):
    client, headers, path, form = conversion_project
    monkeypatch.setattr(
        settings,
        "nb_converter_command",
        [
            PYTHON,
            "-c",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_bytes(b'nb-output')",
            "{output}",
        ],
    )
    conversion_id = uuid4()
    url = progress_path(path, conversion_id)
    response = client.post(
        path,
        headers=tracked_headers(headers, conversion_id),
        data=form,
        files={"file": ("point.onnx", b"onnx-input")},
    )
    assert response.status_code == 200, response.text
    assert client.get(url).status_code == 401
    unknown = client.get(progress_path(path, uuid4()), headers=headers)
    assert unknown.status_code == 404
    assert unknown.headers["cache-control"] == "no-store"

    email = f"progress-{uuid4().hex}@example.com"
    password = "Progress123!"
    created = client.post(
        "/api/v1/users",
        headers=headers,
        json={
            "email": email,
            "password": password,
            "name": "Progress researcher",
            "system_role": "user",
        },
    )
    assert created.status_code == 201, created.text
    project_path = path.removesuffix("/conversions/onnx-to-nb")
    member = client.post(
        f"{project_path}/members", headers=headers, json={"email": email, "role": "researcher"}
    )
    assert member.status_code == 200, member.text
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    other_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    # Project membership alone does not reveal another user's request record.
    assert client.get(url, headers=other_headers).status_code == 404

    other_project = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "name": "Other progress project",
            "project_type": "vibration_feeder_point",
        },
    ).json()
    wrong_project_url = (
        f"/api/v1/projects/{other_project['id']}/conversions/onnx-to-nb/progress/{conversion_id}"
    )
    assert client.get(wrong_project_url, headers=headers).status_code == 404

    for candidate in (headers, other_headers):
        repeated = client.post(
            path,
            headers=tracked_headers(candidate, conversion_id),
            data=form,
            files={"file": ("point.onnx", b"onnx-input")},
        )
        assert repeated.status_code == 409, repeated.text
    original = client.get(url, headers=headers)
    assert original.json()["status"] == "succeeded"
    assert original.json()["percent"] == 100


@pytest.mark.parametrize("invalid_id", ["not-a-uuid", "../../another-progress"])
def test_progress_header_requires_uuid(conversion_project, invalid_id):
    client, headers, path, form = conversion_project
    response = client.post(
        path,
        headers=tracked_headers(headers, invalid_id),
        data=form,
        files={"file": ("point.onnx", b"onnx-input")},
    )
    assert response.status_code == 422, response.text


def test_progress_storage_failure_does_not_prevent_original_conversion(
    conversion_project, monkeypatch
):
    client, headers, path, form = conversion_project
    tasks = Path(settings.nb_converter_workdir) / "converter/tasks"
    tasks.mkdir(exist_ok=True)
    # Simulate unavailable progress storage without breaking the existing
    # writable conversion task directory.
    (tasks / ".progress").write_text("not a directory")
    monkeypatch.setattr(
        settings,
        "nb_converter_command",
        [
            PYTHON,
            "-c",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_bytes(b'nb-output')",
            "{output}",
        ],
    )
    response = client.post(
        path,
        headers=tracked_headers(headers, uuid4()),
        data=form,
        files={"file": ("point.onnx", b"onnx-input")},
    )
    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"


def test_training_artifact_conversion_exposes_same_project_progress(
    conversion_project,
    completed_training,
):
    client, headers, _, _ = conversion_project
    project_path = completed_training["project_path"]
    assert (
        client.patch(
            project_path, headers=headers, json={"project_type": "vibration_feeder_point"}
        ).status_code
        == 200
    )
    run_id, artifact_id = completed_training["run_id"], completed_training["onnx_id"]
    conversion_id = uuid4()
    response = client.post(
        f"/api/v1/runs/{run_id}/artifacts/{artifact_id}/conversions/onnx-to-nb",
        headers=tracked_headers(headers, conversion_id),
    )
    assert response.status_code == 200, response.text
    assert response.content == b"nb-output"
    progress = client.get(
        f"{project_path}/conversions/onnx-to-nb/progress/{conversion_id}",
        headers=headers,
    )
    assert progress.status_code == 200, progress.text
    assert progress.json()["status"] == "succeeded"
    assert progress.json()["percent"] == 100


def test_log_observer_preserves_partial_lines_and_does_not_recount(tmp_path, monkeypatch):
    (tmp_path / "converter").mkdir()
    monkeypatch.setattr(settings, "nb_converter_workdir", str(tmp_path))
    progress = start_progress(uuid4(), "project", "user")
    assert progress is not None
    log = tmp_path / "converter.log"
    log.write_bytes(b"unrelated SDK output\nSDK native stage: inf")
    observer = ConversionLogProgress(log, progress)
    observer.refresh()
    assert progress.data["stage"] == "preparing_dataset"
    with log.open("ab") as file:
        file.write(b"erence\r\n")
    observer.refresh()
    assert progress.data["stage"] == "inference_float"
    first_percent = progress.data["percent"]
    observer.refresh()
    observer.refresh()
    assert progress.data["stage"] == "inference_float"
    with log.open("ab") as file:
        file.write(b"\xff\xfe unreadable text\nSDK stage: inference\n")
    observer.refresh()
    assert progress.data["stage"] == "inference_quantized"
    assert first_percent < progress.data["percent"] < 100
    with log.open("ab") as file:
        file.write(b"SDK native stage: export\n")
    observer.refresh()
    assert progress.data["stage"] == "export"
    assert progress.data["percent"] < 100


def test_cleanup_expires_terminal_records_without_removing_running_tasks(tmp_path, monkeypatch):
    (tmp_path / "converter").mkdir()
    monkeypatch.setattr(settings, "nb_converter_workdir", str(tmp_path))
    succeeded = start_progress(uuid4(), "project", "user")
    failed = start_progress(uuid4(), "project", "user")
    running = start_progress(uuid4(), "project", "user")
    assert succeeded and failed and running
    succeeded.succeed()
    failed.fail("SDK failed")
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    for progress in (succeeded, failed, running):
        data = json.loads(progress.path.read_text(encoding="utf-8"))
        data["updated_at"] = old
        progress.path.write_text(json.dumps(data), encoding="utf-8")
    next_request = start_progress(uuid4(), "project", "user")
    assert next_request is not None
    assert not succeeded.path.exists()
    assert not failed.path.exists()
    assert running.path.is_file()
