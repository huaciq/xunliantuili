from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


class ExecutionStatus(str, Enum):
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class MountSpec:
    source: str
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class ExecutionSpec:
    run_id: str
    image: str
    command: list[str]
    environment: dict[str, str] = field(default_factory=dict)
    gpu_uuids: list[str] = field(default_factory=list)
    mounts: list[MountSpec] = field(default_factory=list)
    workdir: str = "/workspace/code"
    cpu_limit: float = 4.0
    memory_limit_gb: int = 16
    shm_size_gb: int = 4


@dataclass(frozen=True)
class ExecutionHandle:
    execution_id: str
    status: ExecutionStatus
    exit_code: Optional[int] = None


class Executor(ABC):
    @abstractmethod
    def submit(self, spec: ExecutionSpec) -> ExecutionHandle:
        raise NotImplementedError

    @abstractmethod
    def stop(self, execution_id: str, grace_period_seconds: int = 30) -> None:
        raise NotImplementedError

    @abstractmethod
    def inspect(self, execution_id: str) -> ExecutionHandle:
        raise NotImplementedError

    @abstractmethod
    def stream_logs(self, execution_id: str) -> Iterator[str]:
        raise NotImplementedError
