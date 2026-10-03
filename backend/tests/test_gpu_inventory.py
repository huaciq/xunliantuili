from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import GpuDevice, GpuModelPolicy
from app.services import gpu_inventory
from app.services.gpu_inventory import DiscoveredGpu


def isolated_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_refresh_gpu_inventory_allows_indexes_to_swap(monkeypatch) -> None:
    session_factory = isolated_session_factory()
    monkeypatch.setattr(gpu_inventory, "SessionLocal", session_factory)
    with session_factory() as db:
        db.add_all(
            [
                GpuDevice(
                    uuid="GPU-a",
                    index=0,
                    model=GpuModelPolicy.RTX_3090,
                    memory_gb=24,
                ),
                GpuDevice(
                    uuid="GPU-b",
                    index=1,
                    model=GpuModelPolicy.TESLA_T4,
                    memory_gb=15,
                ),
            ]
        )
        db.commit()

    count = gpu_inventory.refresh_gpu_inventory(
        [
            DiscoveredGpu(0, "GPU-b", GpuModelPolicy.TESLA_T4, 15),
            DiscoveredGpu(1, "GPU-a", GpuModelPolicy.RTX_3090, 24),
        ]
    )

    with session_factory() as db:
        devices = list(db.scalars(select(GpuDevice).order_by(GpuDevice.index)).all())
    assert count == 2
    assert [(device.index, device.uuid, device.is_enabled) for device in devices] == [
        (0, "GPU-b", True),
        (1, "GPU-a", True),
    ]


def test_refresh_gpu_inventory_does_not_reuse_an_index_for_new_hardware(monkeypatch) -> None:
    session_factory = isolated_session_factory()
    monkeypatch.setattr(gpu_inventory, "SessionLocal", session_factory)
    with session_factory() as db:
        db.add(
            GpuDevice(
                uuid="GPU-old",
                index=0,
                model=GpuModelPolicy.RTX_3090,
                memory_gb=24,
            )
        )
        db.commit()

    gpu_inventory.refresh_gpu_inventory(
        [DiscoveredGpu(0, "GPU-new", GpuModelPolicy.RTX_4090, 24)]
    )

    with session_factory() as db:
        devices = list(db.scalars(select(GpuDevice).order_by(GpuDevice.index)).all())
    assert [(device.index, device.uuid, device.is_enabled) for device in devices] == [
        (0, "GPU-new", True),
        (1, "GPU-old", False),
    ]
