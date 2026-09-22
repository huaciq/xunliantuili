from pathlib import Path

import pytest

from app.services.scheduler import _resolve_storage_path


def test_resolve_storage_path_joins_relative_cache_uri(tmp_path: Path) -> None:
    cache = tmp_path / "code" / "version-1" / "cache"
    cache.mkdir(parents=True)

    assert _resolve_storage_path(tmp_path, "code/version-1/cache") == cache.resolve()
    assert _resolve_storage_path(tmp_path, str(cache)) == cache.resolve()


def test_resolve_storage_path_rejects_empty_or_escaping_uri(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        _resolve_storage_path(tmp_path, "")

    with pytest.raises(ValueError, match="escapes"):
        _resolve_storage_path(tmp_path, "../outside")
