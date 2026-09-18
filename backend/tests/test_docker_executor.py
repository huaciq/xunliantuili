import json
import subprocess

import pytest

from app.executors import DockerExecutor, ExecutionSpec, ExecutionStatus, MountSpec
from app.services.gpu_inventory import parse_nvidia_smi


class DockerRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.state = {"Status": "running", "Running": True, "ExitCode": 0}

    def __call__(self, argv, **kwargs):
        del kwargs
        self.calls.append(argv)
        if argv[1] == "run":
            return subprocess.CompletedProcess(argv, 0, stdout="container-123\n", stderr="")
        if argv[1] == "inspect":
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.state), stderr="")
        if argv[1] == "logs":
            return subprocess.CompletedProcess(
                argv, 0, stdout="epoch 1\nepoch 2\n", stderr="warn\n"
            )
        return subprocess.CompletedProcess(argv, 0, stdout="container-123\n", stderr="")


def test_docker_executor_builds_hardened_gpu_command_and_maps_state(tmp_path) -> None:
    code = tmp_path / "code"
    output = tmp_path / "output"
    code.mkdir()
    output.mkdir()
    runner = DockerRunner()
    executor = DockerExecutor(runner=runner)
    handle = executor.submit(
        ExecutionSpec(
            run_id="run-1",
            image="training:test",
            command=["python", "train.py"],
            gpu_uuids=["GPU-one", "GPU-two"],
            mounts=[
                MountSpec(str(code), "/workspace/code"),
                MountSpec(str(output), "/workspace/output", read_only=False),
            ],
        )
    )
    command = runner.calls[0]
    assert handle.execution_id == "container-123"
    assert command[command.index("--gpus") + 1] == "device=GPU-one,GPU-two"
    assert "no-new-privileges:true" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert any("target=/workspace/code,readonly" in item for item in command)
    assert any("target=/workspace/output" in item and "readonly" not in item for item in command)
    assert executor.inspect(handle.execution_id).status == ExecutionStatus.RUNNING

    runner.state = {"Status": "exited", "Running": False, "ExitCode": 0}
    assert executor.inspect(handle.execution_id).status == ExecutionStatus.SUCCEEDED
    assert list(executor.stream_logs(handle.execution_id))[-1] == "warn"


def test_docker_executor_rejects_mount_outside_workspace(tmp_path) -> None:
    executor = DockerExecutor(runner=DockerRunner())
    with pytest.raises(ValueError, match="inside /workspace"):
        executor.submit(
            ExecutionSpec(
                run_id="run-1",
                image="training:test",
                command=["python", "train.py"],
                mounts=[MountSpec(str(tmp_path), "/etc/unsafe")],
            )
        )

    with pytest.raises(ValueError, match="cannot run as root"):
        DockerExecutor(user="0:0", runner=DockerRunner())


def test_parse_nvidia_smi_supports_lab_gpu_models() -> None:
    devices = parse_nvidia_smi(
        "0, GPU-a, NVIDIA GeForce RTX 3090, 24576\n"
        "1, GPU-b, NVIDIA GeForce RTX 4090, 24564\n"
        "2, GPU-c, NVIDIA A100, 81920\n"
    )
    assert [(item.index, item.uuid, item.model.value) for item in devices] == [
        (0, "GPU-a", "rtx_3090"),
        (1, "GPU-b", "rtx_4090"),
    ]
