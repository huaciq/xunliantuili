from types import SimpleNamespace

from app.config import settings
from app.models import MetricPoint, RunArtifact
from app.services.results import collect_container_results


class CaptureSession:
    def __init__(self) -> None:
        self.items: list[object] = []

    def scalar(self, statement):
        del statement
        return None

    def add(self, item: object) -> None:
        self.items.append(item)


def test_collect_container_results_reads_ultralytics_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))
    run = SimpleNamespace(id="run-real", project_id="project-real")
    output = tmp_path / "runs" / run.project_id / run.id / "output" / "train"
    weights = output / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"pytorch-checkpoint")
    (output / "results.csv").write_text(
        "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B)\n"
        "0,1.2,0.5,0.4,0.3\n"
        "1,0.8,0.7,0.6,0.55\n",
        encoding="utf-8",
    )
    db = CaptureSession()

    collect_container_results(db, run)

    artifacts = [item for item in db.items if isinstance(item, RunArtifact)]
    metrics = [item for item in db.items if isinstance(item, MetricPoint)]
    assert {item.artifact_type for item in artifacts} == {"checkpoint", "report"}
    assert len(metrics) == 8
    final_map = next(item for item in metrics if item.key == "val/map50" and item.step == 2)
    assert final_map.value == 0.55
