from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal
from app.models import GpuDevice, GpuModelPolicy


@dataclass(frozen=True)
class DiscoveredGpu:
    index: int
    uuid: str
    model: GpuModelPolicy
    memory_gb: int


def parse_nvidia_smi(output: str) -> list[DiscoveredGpu]:
    devices: list[DiscoveredGpu] = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",", 3)]
        if len(parts) != 4:
            continue
        index, uuid, name, memory_mb = parts
        normalized = name.lower().replace("geforce", "")
        if "3090" in normalized:
            model = GpuModelPolicy.RTX_3090
        elif "4090" in normalized:
            model = GpuModelPolicy.RTX_4090
        else:
            continue
        devices.append(
            DiscoveredGpu(
                index=int(index),
                uuid=uuid,
                model=model,
                memory_gb=max(1, round(int(memory_mb) / 1024)),
            )
        )
    return devices


def discover_gpus(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[DiscoveredGpu]:
    try:
        result = runner(
            [
                settings.nvidia_smi_binary,
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"nvidia-smi not found: {settings.nvidia_smi_binary}") from exc
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "nvidia-smi failed").strip())
    return parse_nvidia_smi(result.stdout)


def refresh_gpu_inventory(devices: list[DiscoveredGpu] | None = None) -> int:
    discovered = devices if devices is not None else discover_gpus()
    discovered_uuids = {device.uuid for device in discovered}
    with SessionLocal() as db:
        existing_devices = list(db.scalars(select(GpuDevice)).all())
        by_uuid = {device.uuid: device for device in existing_devices}
        by_index = {device.index: device for device in existing_devices}
        for item in discovered:
            device = by_uuid.get(item.uuid)
            if device is None:
                device = by_index.get(item.index)
            if device is None:
                device = GpuDevice(
                    uuid=item.uuid,
                    index=item.index,
                    model=item.model,
                    memory_gb=item.memory_gb,
                )
                db.add(device)
            else:
                device.uuid = item.uuid
                device.index = item.index
                device.model = item.model
                device.memory_gb = item.memory_gb
                device.is_enabled = True
        for device in existing_devices:
            if device.uuid not in discovered_uuids and not device.uuid.startswith("FAKE-GPU-"):
                device.is_enabled = False
            if device.uuid.startswith("FAKE-GPU-"):
                device.is_enabled = False
        db.commit()
    return len(discovered)
