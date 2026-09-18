from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

from app.executors.base import ExecutionHandle, ExecutionSpec, ExecutionStatus, Executor

Runner = Callable[..., subprocess.CompletedProcess[str]]


class DockerExecutor(Executor):
    """Docker CLI executor with explicit argv and hardened container defaults."""

    def __init__(
        self,
        binary: str = "docker",
        network: str = "bridge",
        user: str = "1000:1000",
        pids_limit: int = 1024,
        runner: Runner = subprocess.run,
    ) -> None:
        if user.split(":", 1)[0] == "0":
            raise ValueError("Docker training containers cannot run as root")
        self.binary = binary
        self.network = network
        self.user = user
        self.pids_limit = pids_limit
        self._runner = runner

    def submit(self, spec: ExecutionSpec) -> ExecutionHandle:
        self._validate_spec(spec)
        argv = [
            self.binary,
            "run",
            "--detach",
            "--name",
            f"train-platform-{spec.run_id}",
            "--label",
            "train-platform.managed=true",
            "--label",
            f"train-platform.run-id={spec.run_id}",
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--init",
            "--user",
            self.user,
            "--pids-limit",
            str(self.pids_limit),
            "--cpus",
            str(spec.cpu_limit),
            "--memory",
            f"{spec.memory_limit_gb}g",
            "--shm-size",
            f"{spec.shm_size_gb}g",
            "--workdir",
            spec.workdir,
        ]
        if self.network:
            argv.extend(["--network", self.network])
        if spec.gpu_uuids:
            argv.extend(["--gpus", f"device={','.join(spec.gpu_uuids)}"])
        for key, value in sorted(spec.environment.items()):
            argv.extend(["--env", f"{key}={value}"])
        for mount in spec.mounts:
            option = f"type=bind,source={Path(mount.source).resolve()},target={mount.target}"
            if mount.read_only:
                option += ",readonly"
            argv.extend(["--mount", option])
        argv.append(spec.image)
        argv.extend(spec.command)
        result = self._run(argv)
        execution_id = result.stdout.strip()
        if not execution_id:
            raise RuntimeError("Docker did not return a container id")
        return ExecutionHandle(execution_id=execution_id, status=ExecutionStatus.RUNNING)

    def stop(self, execution_id: str, grace_period_seconds: int = 30) -> None:
        self._run(
            [self.binary, "stop", "--time", str(grace_period_seconds), execution_id],
            missing_is_lookup=True,
        )

    def inspect(self, execution_id: str) -> ExecutionHandle:
        result = self._run(
            [self.binary, "inspect", "--format", "{{json .State}}", execution_id],
            missing_is_lookup=True,
        )
        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Docker returned invalid container state") from exc
        status = str(state.get("Status", ""))
        exit_code = state.get("ExitCode")
        if bool(state.get("Running")) or status in {"created", "restarting"}:
            mapped = ExecutionStatus.RUNNING
            exit_code = None
        elif status in {"exited", "dead"} and exit_code == 0:
            mapped = ExecutionStatus.SUCCEEDED
        elif status in {"exited", "dead"} and exit_code in {137, 143}:
            mapped = ExecutionStatus.STOPPED
        else:
            mapped = ExecutionStatus.FAILED
        return ExecutionHandle(execution_id=execution_id, status=mapped, exit_code=exit_code)

    def stream_logs(self, execution_id: str) -> Iterator[str]:
        result = self._run(
            [self.binary, "logs", "--tail", "5000", execution_id],
            missing_is_lookup=True,
        )
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        yield from output.splitlines()

    def _run(
        self, argv: list[str], missing_is_lookup: bool = False
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(argv, capture_output=True, text=True, timeout=60, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Docker CLI not found: {self.binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Docker command timed out") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if missing_is_lookup and "No such" in detail:
                raise LookupError(detail)
            message = detail or f"Docker command failed with exit code {result.returncode}"
            raise RuntimeError(message)
        return result

    @staticmethod
    def _validate_spec(spec: ExecutionSpec) -> None:
        if not spec.image.strip() or not spec.command:
            raise ValueError("Image and command are required")
        workdir = PurePosixPath(spec.workdir)
        if not workdir.is_absolute() or not workdir.is_relative_to(PurePosixPath("/workspace")):
            raise ValueError("Container workdir must be inside /workspace")
        targets: set[str] = set()
        for mount in spec.mounts:
            source = Path(mount.source).resolve()
            target = PurePosixPath(mount.target)
            if not source.exists():
                raise ValueError(f"Mount source does not exist: {source}")
            if not target.is_absolute() or not target.is_relative_to(PurePosixPath("/workspace")):
                raise ValueError("Mount targets must be inside /workspace")
            if str(target) in targets:
                raise ValueError(f"Duplicate mount target: {target}")
            targets.add(str(target))
