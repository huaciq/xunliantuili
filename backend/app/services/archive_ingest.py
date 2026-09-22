from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.models import DatasetFormat

ALLOWED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class IngestResult:
    archive_uri: str
    cache_uri: str
    manifest_uri: str
    sha256: str
    archive_size: int
    extracted_size: int
    file_count: int
    dataset_format: DatasetFormat = DatasetFormat.GENERIC
    dataset_root_subpath: str = "."


def storage_root() -> Path:
    root = Path(settings.storage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_archive_filename(filename: str) -> str:
    safe_name = Path(filename or "upload.zip").name
    lowered = safe_name.lower()
    if not any(lowered.endswith(suffix) for suffix in ALLOWED_ARCHIVE_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP, TAR, TAR.GZ and TGZ archives are supported",
        )
    return safe_name


async def save_upload_to_temp(upload: UploadFile) -> tuple[Path, str, int, str]:
    filename = validate_archive_filename(upload.filename or "")
    suffix = ".tar.gz" if filename.lower().endswith((".tar.gz", ".tgz")) else Path(filename).suffix
    digest = hashlib.sha256()
    total = 0
    temp = tempfile.NamedTemporaryFile(prefix="train-platform-", suffix=suffix, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Archive exceeds the {settings.max_upload_bytes} byte limit",
                    )
                digest.update(chunk)
                temp.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return temp_path, digest.hexdigest(), total, filename


def ingest_archive(
    temp_path: Path,
    destination_prefix: Path,
    filename: str,
    sha256: str,
    archive_size: int,
    detect_dataset: bool,
) -> IngestResult:
    root = storage_root()
    target = (root / destination_prefix).resolve()
    if not _is_within(root, target):
        raise ValueError("Storage destination escaped the configured root")

    archive_dir = target / "source"
    cache_dir = target / "cache"
    manifest_path = target / "manifest.json"
    archive_path = archive_dir / filename
    target.mkdir(parents=True, exist_ok=False)
    archive_dir.mkdir()
    cache_dir.mkdir()

    try:
        shutil.move(str(temp_path), archive_path)
        _extract_archive(archive_path, cache_dir)
        entries, extracted_size = _build_manifest(cache_dir)
        detected, dataset_root_subpath = (
            _detect_dataset_format(cache_dir, entries)
            if detect_dataset
            else (DatasetFormat.GENERIC, ".")
        )
        manifest = {
            "schema_version": 1,
            "source_filename": filename,
            "sha256": sha256,
            "archive_size": archive_size,
            "extracted_size": extracted_size,
            "file_count": len(entries),
            "dataset_format": detected.value if detect_dataset else None,
            "dataset_root_subpath": dataset_root_subpath if detect_dataset else None,
            "files": entries,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return IngestResult(
            archive_uri=_relative_uri(root, archive_path),
            cache_uri=_relative_uri(root, cache_dir),
            manifest_uri=_relative_uri(root, manifest_path),
            sha256=sha256,
            archive_size=archive_size,
            extracted_size=extracted_size,
            file_count=len(entries),
            dataset_format=detected,
            dataset_root_subpath=dataset_root_subpath,
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        temp_path.unlink(missing_ok=True)
        raise


def _relative_uri(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_within(root: Path, candidate: Path) -> bool:
    root_value = os.path.normcase(str(root.resolve()))
    candidate_value = os.path.normcase(str(candidate.resolve()))
    try:
        return os.path.commonpath([root_value, candidate_value]) == root_value
    except ValueError:
        return False


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe archive path: {member_name}")
    destination = (root / Path(*member.parts)).resolve()
    if not _is_within(root, destination):
        raise ValueError(f"Archive path escaped extraction root: {member_name}")
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> None:
    lowered = archive_path.name.lower()
    if lowered.endswith(".zip"):
        _extract_zip(archive_path, destination)
    else:
        _extract_tar(archive_path, destination)


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        _validate_archive_totals(len(members), sum(member.file_size for member in members))
        for member in members:
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed: {member.filename}")
            target = _safe_destination(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        _validate_archive_totals(len(members), sum(member.size for member in members))
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Links and device files are not allowed: {member.name}")
            target = _safe_destination(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _validate_archive_totals(entry_count: int, extracted_size: int) -> None:
    if entry_count > settings.max_archive_entries:
        raise ValueError(f"Archive contains more than {settings.max_archive_entries} entries")
    if extracted_size > settings.max_extracted_bytes:
        raise ValueError(f"Extracted archive exceeds {settings.max_extracted_bytes} bytes")


def _build_manifest(root: Path) -> tuple[list[dict[str, object]], int]:
    entries: list[dict[str, object]] = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total += size
        entries.append({"path": path.relative_to(root).as_posix(), "size": size})
    return entries, total


def _detect_dataset_format(
    root: Path, entries: list[dict[str, object]]
) -> tuple[DatasetFormat, str]:
    paths = [str(entry["path"]).lower() for entry in entries]
    names = {Path(path).name for path in paths}
    if "data.yaml" in names or "data.yml" in names:
        return DatasetFormat.YOLO, "."
    mvtec_root = _find_mvtec_root(root)
    if mvtec_root is not None:
        relative = mvtec_root.relative_to(root).as_posix()
        return DatasetFormat.MVTEC_AD, relative or "."
    if any(path.endswith(".json") and "annotation" in path for path in paths):
        try:
            for json_path in root.rglob("*.json"):
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                if (
                    isinstance(payload, dict)
                    and {"images", "annotations", "categories"} <= payload.keys()
                ):
                    return DatasetFormat.COCO, "."
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_parents = {
        Path(str(entry["path"])).parent.as_posix()
        for entry in entries
        if Path(str(entry["path"])).suffix.lower() in image_suffixes
    }
    if len(image_parents) >= 2 and not any("/labels/" in f"/{path}/" for path in paths):
        return DatasetFormat.CLASSIFICATION, "."
    return DatasetFormat.GENERIC, "."


def _find_mvtec_root(root: Path) -> Path | None:
    """Locate an MVTec AD root, including archives wrapped in one or more folders."""
    candidates = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for candidate in candidates:
        category_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        valid_categories = [
            path
            for path in category_dirs
            if (path / "train").is_dir()
            and (path / "test").is_dir()
            and (path / "ground_truth").is_dir()
        ]
        if valid_categories:
            return candidate
    return None
