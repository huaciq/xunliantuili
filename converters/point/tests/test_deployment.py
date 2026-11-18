import importlib.util
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/deploy_point_converter.py"
spec = importlib.util.spec_from_file_location("deploy_point_converter", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_environment_update_selects_only_conversion_settings():
    original = "APP_SECRET_KEY=keep-me\nNB_CONVERTER_WORKDIR=/opt/old\nNB_CONVERSION_TIMEOUT_SECONDS=600\n"
    result = installer.update_environment(original, PurePosixPath("/opt/glh-converter-zys"))
    assert "APP_SECRET_KEY=keep-me" in result
    assert 'NB_CONVERTER_WORKDIR="/opt/glh-converter-zys"' in result
    assert "NB_CONVERSION_TIMEOUT_SECONDS=1800" in result
    assert '"{input}"' in result and '"{dataset}"' in result and '"{output}"' in result
    assert installer.update_environment(result, PurePosixPath("/opt/glh-converter-zys")) == result


def test_environment_update_preserves_longer_timeouts_and_removes_duplicate_keys():
    result = installer.update_environment(
        "NB_CONVERSION_TIMEOUT_SECONDS='2400'\nNB_CONVERTER_WORKDIR=/opt/old\n"
        "export NB_CONVERTER_WORKDIR=/opt/duplicate\nOTHER=unchanged\n",
        PurePosixPath("/opt/glh-converter-zys"),
    )
    assert "NB_CONVERSION_TIMEOUT_SECONDS=2400" in result
    assert result.count("NB_CONVERTER_WORKDIR=") == 1
    assert "OTHER=unchanged" in result


@pytest.mark.parametrize("import_failure", [False, True])
def test_deployment_selects_new_version_only_after_successful_import(tmp_path, monkeypatch, import_failure):
    target = tmp_path / "installed"
    modules = target / "converter"
    modules.mkdir(parents=True)
    (modules / "convert_nb.py").write_text("old version\n")
    for name in installer.RETIRED_MODULES:
        (modules / name).write_text("previous added module\n")
    env_file = tmp_path / "platform.env"
    env_file.write_text("NB_CONVERTER_WORKDIR=/opt/old\nAPP_SECRET_KEY=retain\n")
    original = env_file.read_bytes()
    owner = SimpleNamespace(pw_uid=1000, pw_gid=1000)
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=lambda name: owner))
    monkeypatch.setattr(installer.os, "chown", lambda *args: None, raising=False)
    real_run = subprocess.run

    def execute(argv, **kwargs):
        if "-B" in argv:
            if import_failure:
                raise subprocess.CalledProcessError(1, argv)
            return real_run([sys.executable, "-B", "-c", argv[-1]], **kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(installer.subprocess, "run", execute)
    args = SimpleNamespace(converter_root=target, seed_root=tmp_path / "seed",
                           env_file=env_file, user="huaciq")
    if import_failure:
        with pytest.raises(subprocess.CalledProcessError):
            installer.deploy(args)
        assert env_file.read_bytes() == original
        assert (modules / "convert_nb.py").read_text() == "old version\n"
        assert all((modules / name).read_text() == "previous added module\n"
                   for name in installer.RETIRED_MODULES)
        assert not (modules / "original_source.json").exists()
    else:
        installer.deploy(args)
        assert 'APP_SECRET_KEY=retain' in env_file.read_text()
        assert str(target).replace('\\', '\\\\') in env_file.read_text()
        assert (modules / "original_source.json").is_file()
        assert not any((modules / name).exists() for name in installer.RETIRED_MODULES)
