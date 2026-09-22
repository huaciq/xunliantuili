from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path, PurePosixPath

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import SessionLocal
from app.executors import DockerExecutor, ExecutionSpec, ExecutionStatus, FakeExecutor, MountSpec
from app.models import (
    GpuAllocation,
    GpuDevice,
    GpuModelPolicy,
    RunEvent,
    TrainingRun,
    TrainingRunStatus,
    utc_now,
)
from app.services.results import collect_run_results

ACTIVE_STATUSES = {
    TrainingRunStatus.PREPARING,
    TrainingRunStatus.STARTING,
    TrainingRunStatus.RUNNING,
    TrainingRunStatus.CANCEL_REQUESTED,
    TrainingRunStatus.STOPPING,
}

logger = logging.getLogger(__name__)


def _resolve_storage_path(storage_root: Path, uri: str) -> Path:
    """Resolve a persisted storage URI without allowing it to escape STORAGE_ROOT."""
    if not uri.strip():
        raise ValueError("Storage URI is empty")
    raw_path = Path(uri)
    candidate = raw_path if raw_path.is_absolute() else storage_root / raw_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(storage_root):
        raise ValueError(f"Storage URI escapes the configured root: {uri}")
    return resolved


class TrainingScheduler:
    def __init__(self) -> None:
        if settings.executor_backend == "docker":
            self.executor = DockerExecutor(
                binary=settings.docker_binary,
                network=settings.docker_network,
                user=settings.docker_user,
                pids_limit=settings.docker_pids_limit,
            )
        else:
            self.executor = FakeExecutor(settings.fake_run_duration_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._recover_interrupted_runs()
        self._thread = threading.Thread(target=self._loop, name="training-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    def logs(self, execution_id: str | None) -> list[str]:
        if not execution_id:
            return []
        try:
            return list(self.executor.stream_logs(execution_id))
        except LookupError:
            return ["Execution state is unavailable after scheduler restart"]

    def _loop(self) -> None:
        while not self._stop_event.wait(settings.scheduler_poll_seconds):
            try:
                self._poll_active_runs()
                self._schedule_queued_runs()
            except Exception:
                # Keep the control loop alive; individual run failures are recorded where handled.
                continue

    def _recover_interrupted_runs(self) -> None:
        with SessionLocal() as db:
            runs = db.scalars(
                select(TrainingRun).where(TrainingRun.status.in_(ACTIVE_STATUSES))
            ).all()
            for run in runs:
                if isinstance(self.executor, FakeExecutor):
                    self._finish_run(
                        db,
                        run,
                        TrainingRunStatus.FAILED,
                        None,
                        "Scheduler restarted while the development executor was active",
                    )
                    continue
                if not run.execution_id:
                    self._finish_run(
                        db,
                        run,
                        TrainingRunStatus.FAILED,
                        None,
                        "Active run had no Docker container id during reconciliation",
                    )
                    continue
                try:
                    handle = self.executor.inspect(run.execution_id)
                except LookupError:
                    self._finish_run(
                        db,
                        run,
                        TrainingRunStatus.FAILED,
                        None,
                        "Docker container disappeared during scheduler restart",
                    )
                    continue
                except RuntimeError:
                    continue
                if handle.status == ExecutionStatus.RUNNING:
                    run.status = TrainingRunStatus.RUNNING
                    db.add(
                        RunEvent(
                            run_id=run.id,
                            event_type="reconcile",
                            message="调度器重启后已重新关联运行中的 Docker 容器",
                        )
                    )
                elif handle.status == ExecutionStatus.SUCCEEDED:
                    self._finish_run(db, run, TrainingRunStatus.SUCCEEDED, handle.exit_code)
                elif handle.status == ExecutionStatus.STOPPED:
                    self._finish_run(db, run, TrainingRunStatus.STOPPED, handle.exit_code)
                else:
                    self._finish_run(db, run, TrainingRunStatus.FAILED, handle.exit_code)
            db.commit()

    def _schedule_queued_runs(self) -> None:
        with SessionLocal() as db:
            runs = db.scalars(
                select(TrainingRun)
                .where(TrainingRun.status == TrainingRunStatus.QUEUED)
                .options(
                    selectinload(TrainingRun.runtime_image),
                    selectinload(TrainingRun.dataset_version),
                    selectinload(TrainingRun.code_version),
                )
                .order_by(TrainingRun.priority.desc(), TrainingRun.queued_at.asc())
            ).all()
            for run in runs:
                devices = self._select_gpus(db, run)
                if devices is None:
                    continue
                try:
                    run.status = TrainingRunStatus.PREPARING
                    for device in devices:
                        db.add(GpuAllocation(run_id=run.id, gpu_device_id=device.id))
                    db.flush()
                    handle = self.executor.submit(self._execution_spec(run, devices))
                    run.execution_id = handle.execution_id
                    run.status = TrainingRunStatus.RUNNING
                    run.started_at = utc_now()
                    gpu_text = ", ".join(device.model.value for device in devices) or "CPU"
                    db.add(
                        RunEvent(
                            run_id=run.id,
                            event_type="status",
                            message=f"任务已启动，资源：{gpu_text}",
                        )
                    )
                    db.commit()
                except Exception as exc:
                    logger.exception("Failed to start training run %s", run.id)
                    db.rollback()
                    failed = db.get(TrainingRun, run.id)
                    if failed:
                        failed.status = TrainingRunStatus.FAILED
                        failed.failure_reason = str(exc)[:2000]
                        failed.finished_at = utc_now()
                        db.add(
                            RunEvent(
                                run_id=failed.id,
                                event_type="error",
                                message=f"训练任务启动失败：{failed.failure_reason}",
                            )
                        )
                        db.commit()

    def _poll_active_runs(self) -> None:
        with SessionLocal() as db:
            runs = db.scalars(
                select(TrainingRun).where(TrainingRun.status.in_(ACTIVE_STATUSES))
            ).all()
            for run in runs:
                if run.status == TrainingRunStatus.CANCEL_REQUESTED:
                    run.status = TrainingRunStatus.STOPPING
                    if run.execution_id:
                        try:
                            self.executor.stop(run.execution_id)
                        except LookupError:
                            pass
                    self._finish_run(db, run, TrainingRunStatus.STOPPED, 143)
                    continue
                if not run.execution_id:
                    continue
                try:
                    handle = self.executor.inspect(run.execution_id)
                except LookupError:
                    self._finish_run(
                        db,
                        run,
                        TrainingRunStatus.FAILED,
                        None,
                        "Execution disappeared from the development executor",
                    )
                    continue
                if handle.status == ExecutionStatus.SUCCEEDED:
                    self._finish_run(db, run, TrainingRunStatus.SUCCEEDED, handle.exit_code)
                elif handle.status == ExecutionStatus.FAILED:
                    self._finish_run(db, run, TrainingRunStatus.FAILED, handle.exit_code)
                elif handle.status == ExecutionStatus.STOPPED:
                    self._finish_run(db, run, TrainingRunStatus.STOPPED, handle.exit_code)
            db.commit()

    def _finish_run(
        self,
        db,
        run: TrainingRun,
        status: TrainingRunStatus,
        exit_code: int | None,
        failure_reason: str = "",
    ) -> None:
        run.status = status
        run.exit_code = exit_code
        run.failure_reason = failure_reason
        run.finished_at = utc_now()
        if status == TrainingRunStatus.SUCCEEDED:
            try:
                collect_run_results(db, run, development=isinstance(self.executor, FakeExecutor))
            except Exception as exc:
                db.add(
                    RunEvent(
                        run_id=run.id,
                        event_type="artifact",
                        message=f"训练成功，但产物采集失败：{str(exc)[:500]}",
                    )
                )
        db.execute(delete(GpuAllocation).where(GpuAllocation.run_id == run.id))
        db.add(
            RunEvent(run_id=run.id, event_type="status", message=f"任务状态变更为 {status.value}")
        )

    def _select_gpus(self, db, run: TrainingRun) -> list[GpuDevice] | None:
        if run.requested_gpu_count == 0:
            return []
        allocated_ids = set(db.scalars(select(GpuAllocation.gpu_device_id)).all())
        devices = list(
            db.scalars(
                select(GpuDevice)
                .where(GpuDevice.is_enabled.is_(True), GpuDevice.id.not_in(allocated_ids))
                .order_by(GpuDevice.index)
            ).all()
        )
        if run.requested_gpu_model != GpuModelPolicy.ANY:
            devices = [device for device in devices if device.model == run.requested_gpu_model]
            return (
                devices[: run.requested_gpu_count]
                if len(devices) >= run.requested_gpu_count
                else None
            )

        groups: dict[GpuModelPolicy, list[GpuDevice]] = defaultdict(list)
        for device in devices:
            groups[device.model].append(device)
        candidates = [group for group in groups.values() if len(group) >= run.requested_gpu_count]
        if not candidates:
            return None
        candidates.sort(key=lambda group: (len(group), group[0].index))
        return candidates[0][: run.requested_gpu_count]

    def _execution_spec(self, run: TrainingRun, devices: list[GpuDevice]) -> ExecutionSpec:
        if isinstance(self.executor, FakeExecutor):
            return ExecutionSpec(
                run_id=run.id,
                image=run.runtime_image.image,
                command=run.command,
                environment={"PLATFORM_RUN_ID": run.id},
                gpu_uuids=[device.uuid for device in devices],
            )

        storage_root = Path(settings.storage_root).resolve()
        code_root = _resolve_storage_path(storage_root, run.code_version.cache_uri)
        dataset_root = _resolve_storage_path(storage_root, run.dataset_version.cache_uri)
        run_root = storage_root / "runs" / run.project_id / run.id
        output_root = run_root / "output"
        config_root = run_root / "config"
        output_root.mkdir(parents=True, exist_ok=True)
        config_root.mkdir(parents=True, exist_ok=True)
        config = {
            "run_id": run.id,
            "project_id": run.project_id,
            "parameters": run.parameters,
            "command": run.command,
            "dataset_version_id": run.dataset_version_id,
            "code_version_id": run.code_version_id,
            "gpu_uuids": [device.uuid for device in devices],
        }
        (config_root / "run.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        relative_workdir = PurePosixPath(run.code_version.default_workdir or ".")
        if relative_workdir.is_absolute() or ".." in relative_workdir.parts:
            raise ValueError("Code workdir must be a relative path without parent traversal")
        workdir = str(PurePosixPath("/workspace/code") / relative_workdir)
        return ExecutionSpec(
            run_id=run.id,
            image=run.runtime_image.image,
            command=run.command,
            environment={
                "PLATFORM_RUN_ID": run.id,
                "PLATFORM_OUTPUT_DIR": "/workspace/output",
            },
            gpu_uuids=[device.uuid for device in devices],
            mounts=[
                MountSpec(str(code_root), "/workspace/code", read_only=True),
                MountSpec(str(dataset_root), "/workspace/dataset", read_only=True),
                MountSpec(str(output_root), "/workspace/output", read_only=False),
                MountSpec(str(config_root), "/workspace/config", read_only=True),
            ],
            workdir=workdir,
            cpu_limit=settings.docker_cpu_limit,
            memory_limit_gb=settings.docker_memory_limit_gb,
            shm_size_gb=settings.docker_shm_size_gb,
        )


scheduler = TrainingScheduler()
