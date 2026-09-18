import time
from dataclasses import dataclass
from threading import Lock
from typing import Iterator

from app.executors.base import ExecutionHandle, ExecutionSpec, ExecutionStatus, Executor


@dataclass
class FakeExecution:
    handle: ExecutionHandle
    started_at: float
    duration_seconds: int
    image: str
    command: list[str]


class FakeExecutor(Executor):
    """In-memory executor for control-plane development without Docker or GPUs."""

    def __init__(self, duration_seconds: int = 12) -> None:
        self.duration_seconds = duration_seconds
        self._executions: dict[str, FakeExecution] = {}
        self._lock = Lock()

    def submit(self, spec: ExecutionSpec) -> ExecutionHandle:
        handle = ExecutionHandle(execution_id=f"fake-{spec.run_id}", status=ExecutionStatus.RUNNING)
        with self._lock:
            self._executions[handle.execution_id] = FakeExecution(
                handle=handle,
                started_at=time.monotonic(),
                duration_seconds=self.duration_seconds,
                image=spec.image,
                command=spec.command,
            )
        return handle

    def stop(self, execution_id: str, grace_period_seconds: int = 30) -> None:
        del grace_period_seconds
        with self._lock:
            execution = self._require(execution_id)
            execution.handle = ExecutionHandle(
                execution_id=execution.handle.execution_id,
                status=ExecutionStatus.STOPPED,
                exit_code=143,
            )

    def inspect(self, execution_id: str) -> ExecutionHandle:
        with self._lock:
            execution = self._require(execution_id)
            self._refresh(execution)
            return execution.handle

    def stream_logs(self, execution_id: str) -> Iterator[str]:
        with self._lock:
            execution = self._require(execution_id)
            self._refresh(execution)
            elapsed = min(time.monotonic() - execution.started_at, execution.duration_seconds)
            progress = int(elapsed / max(execution.duration_seconds, 1) * 100)
            lines = [
                f"Preparing image {execution.image}",
                f"Starting command: {' '.join(execution.command)}",
            ]
            milestones = [20, 40, 60, 80, 100]
            for milestone in milestones:
                if progress >= milestone:
                    epoch = milestone // 20
                    lines.append(
                        f"epoch={epoch}/5 progress={milestone}% loss={1.0 / (epoch + 1):.4f}"
                    )
            if execution.handle.status == ExecutionStatus.SUCCEEDED:
                lines.append("Training completed successfully")
            elif execution.handle.status == ExecutionStatus.STOPPED:
                lines.append("Execution stopped")
        yield from lines

    def _refresh(self, execution: FakeExecution) -> None:
        if execution.handle.status != ExecutionStatus.RUNNING:
            return
        if time.monotonic() - execution.started_at >= execution.duration_seconds:
            execution.handle = ExecutionHandle(
                execution_id=execution.handle.execution_id,
                status=ExecutionStatus.SUCCEEDED,
                exit_code=0,
            )

    def _require(self, execution_id: str) -> FakeExecution:
        try:
            return self._executions[execution_id]
        except KeyError as exc:
            raise LookupError(f"Unknown execution: {execution_id}") from exc
