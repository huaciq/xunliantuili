from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.models import MetricPoint, RunArtifact, TrainingRun

COLLECTED_EXTENSIONS = {".pt", ".pth", ".onnx", ".csv", ".json", ".yaml", ".yml", ".png"}


def _write_artifact(path: Path, content: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


def collect_development_results(db, run: TrainingRun) -> None:
    """Create deterministic development results for the fake executor."""
    existing = db.scalar(select(RunArtifact.id).where(RunArtifact.run_id == run.id).limit(1))
    if existing:
        return

    root = Path(settings.storage_root).resolve() / "runs" / run.project_id / run.id / "artifacts"
    epochs = max(1, min(int(run.parameters.get("epochs", 5)), 20))
    points: list[MetricPoint] = []
    rows: list[dict[str, float | int]] = []
    for step in range(1, epochs + 1):
        progress = step / epochs
        values = {
            "train/loss": round(1.15 / (step + 0.7), 5),
            "val/precision": round(0.48 + 0.43 * progress, 5),
            "val/recall": round(0.42 + 0.46 * progress, 5),
            "val/map50": round(0.35 + 0.57 * progress, 5),
        }
        rows.append({"epoch": step, **values})
        points.extend(
            MetricPoint(run_id=run.id, key=key, step=step, value=value)
            for key, value in values.items()
        )
    db.add_all(points)

    checkpoint = root / "best.pt"
    checkpoint_content = json.dumps(
        {
            "development_artifact": True,
            "run_id": run.id,
            "template": run.template.key,
            "parameters": run.parameters,
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode()
    size, digest = _write_artifact(checkpoint, checkpoint_content)
    db.add(
        RunArtifact(
            run_id=run.id,
            artifact_type="checkpoint",
            name="best.pt",
            uri=str(checkpoint),
            size_bytes=size,
            sha256=digest,
            details={"format": "pytorch", "development_artifact": True},
        )
    )

    report = root / "metrics.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_content = report.read_bytes()
    db.add(
        RunArtifact(
            run_id=run.id,
            artifact_type="report",
            name="metrics.csv",
            uri=str(report),
            size_bytes=len(report_content),
            sha256=hashlib.sha256(report_content).hexdigest(),
            details={"rows": epochs},
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_key(column: str) -> str | None:
    key = column.strip()
    lower = key.lower()
    if lower in {"epoch", "step", "time"}:
        return None
    if "map50-95" in lower:
        return "val/map50-95"
    if "map50" in lower:
        return "val/map50"
    if "precision" in lower:
        return "val/precision"
    if "recall" in lower:
        return "val/recall"
    if "loss" in lower:
        return key.replace("metrics/", "val/")
    return None


def _collect_metrics_csv(db, run: TrainingRun, path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            raw_step = row.get("epoch") or row.get(" step") or str(row_index)
            try:
                step = int(float(raw_step)) + (1 if "epoch" in row else 0)
            except (TypeError, ValueError):
                step = row_index
            for column, raw_value in row.items():
                metric_key = _metric_key(column)
                if metric_key is None:
                    continue
                try:
                    value = float(str(raw_value).strip())
                except (TypeError, ValueError):
                    continue
                db.add(MetricPoint(run_id=run.id, key=metric_key, step=step, value=value))


def collect_container_results(db, run: TrainingRun) -> None:
    existing = db.scalar(select(RunArtifact.id).where(RunArtifact.run_id == run.id).limit(1))
    if existing:
        return
    output_root = (
        Path(settings.storage_root).resolve() / "runs" / run.project_id / run.id / "output"
    )
    if not output_root.is_dir():
        return

    candidates: list[Path] = []
    for path in sorted(output_root.rglob("*")):
        resolved = path.resolve()
        if (
            len(candidates) >= settings.max_collected_artifacts
            or path.is_symlink()
            or not path.is_file()
            or not resolved.is_relative_to(output_root)
            or path.suffix.lower() not in COLLECTED_EXTENSIONS
        ):
            continue
        candidates.append(path)

    total_bytes = 0
    metrics_collected = False
    for path in candidates:
        size = path.stat().st_size
        if total_bytes + size > settings.max_collected_artifact_bytes:
            continue
        total_bytes += size
        suffix = path.suffix.lower()
        relative_name = path.relative_to(output_root).as_posix()
        if suffix in {".pt", ".pth"}:
            artifact_type = "checkpoint"
            artifact_format = "pytorch"
        elif suffix == ".onnx":
            artifact_type = "export"
            artifact_format = "onnx"
        else:
            artifact_type = "report"
            artifact_format = suffix.lstrip(".")
        db.add(
            RunArtifact(
                run_id=run.id,
                artifact_type=artifact_type,
                name=relative_name,
                uri=str(path.resolve()),
                size_bytes=size,
                sha256=_sha256_file(path),
                details={"format": artifact_format, "development_artifact": False},
            )
        )
        if suffix == ".csv" and path.name.lower() == "results.csv" and not metrics_collected:
            _collect_metrics_csv(db, run, path)
            metrics_collected = True


def collect_run_results(db, run: TrainingRun, development: bool) -> None:
    if development:
        collect_development_results(db, run)
    else:
        collect_container_results(db, run)
