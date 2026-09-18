from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal
from app.models import GpuDevice, GpuModelPolicy, RuntimeImage, SystemRole, TrainingTemplate, User
from app.security import hash_password


def ensure_bootstrap_admin() -> None:
    with SessionLocal() as db:
        existing = db.scalar(
            select(User).where(User.email == settings.bootstrap_admin_email.lower())
        )
        if existing:
            return
        db.add(
            User(
                email=settings.bootstrap_admin_email.lower(),
                name=settings.bootstrap_admin_name,
                password_hash=hash_password(settings.bootstrap_admin_password),
                system_role=SystemRole.ADMIN,
            )
        )
        db.commit()


def ensure_development_catalog() -> None:
    with SessionLocal() as db:
        runtime = db.scalar(select(RuntimeImage).where(RuntimeImage.name == "Ultralytics YOLO"))
        if runtime is None:
            runtime = RuntimeImage(
                name="Ultralytics YOLO",
                image="train-platform/ultralytics:development",
                digest="development",
                framework="ultralytics",
            )
            db.add(runtime)
            db.flush()
        custom_runtime = db.scalar(select(RuntimeImage).where(RuntimeImage.name == "PyTorch Base"))
        if custom_runtime is None:
            custom_runtime = RuntimeImage(
                name="PyTorch Base",
                image="train-platform/pytorch:development",
                digest="development",
                framework="pytorch",
            )
            db.add(custom_runtime)
            db.flush()

        templates = {
            "yolo_detection": (
                "YOLO 目标检测",
                "使用 Ultralytics YOLO 训练目标检测模型。",
                runtime.id,
                {
                    "epochs": {"type": "integer", "default": 100, "minimum": 1},
                    "batch": {"type": "integer", "default": 16, "minimum": 1},
                    "imgsz": {"type": "integer", "default": 640, "minimum": 32},
                    "model": {"type": "string", "default": "yolo11n.pt"},
                },
            ),
            "custom_command": (
                "自定义命令",
                "运行代码版本中定义的入口命令。",
                custom_runtime.id,
                {"arguments": {"type": "string", "default": ""}},
            ),
        }
        for key, (name, description, runtime_id, schema) in templates.items():
            if db.scalar(select(TrainingTemplate).where(TrainingTemplate.key == key)) is None:
                db.add(
                    TrainingTemplate(
                        key=key,
                        name=name,
                        description=description,
                        default_runtime_image_id=runtime_id,
                        parameter_schema=schema,
                    )
                )

        if settings.executor_backend == "fake":
            gpu_specs = [
                (0, GpuModelPolicy.RTX_3090, 24),
                (1, GpuModelPolicy.RTX_3090, 24),
                (2, GpuModelPolicy.RTX_4090, 24),
                (3, GpuModelPolicy.RTX_4090, 24),
            ]
            for index, model, memory_gb in gpu_specs:
                uuid = f"FAKE-GPU-{index}-{model.value.upper()}"
                if db.scalar(select(GpuDevice).where(GpuDevice.uuid == uuid)) is None:
                    db.add(GpuDevice(uuid=uuid, index=index, model=model, memory_gb=memory_gb))
        db.commit()
