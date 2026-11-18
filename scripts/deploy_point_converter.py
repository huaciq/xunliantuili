"""Install the checked point converter and select it for one platform instance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "converters" / "point"
MODULES = (
    "convert_nb.py", "convert_nb_docker.py", "point_quantization.py",
    "nb_validation.py", "hybrid_quantization.py", "sdk_point_entry.py", "sdk_container_watchdog.py",
)


def replace_file(source, destination):
    """Publish a complete file while retaining ownership and access permissions."""
    previous = destination.stat() if destination.exists() else None
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        shutil.copyfile(source, staged)
        staged.chmod(previous.st_mode & 0o777 if previous else 0o644)
        if previous:
            os.chown(staged, previous.st_uid, previous.st_gid)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def update_environment(contents, converter_root):
    command = ["/bin/bash", "./convert.sh", "{input}", "{dataset}", "{output}"]
    updates = {
        "NB_CONVERTER_WORKDIR": json.dumps(str(converter_root)),
        "NB_CONVERTER_COMMAND": "'" + json.dumps(command) + "'",
        "NB_CONVERSION_TIMEOUT_SECONDS": "1800",
    }
    lines = []
    seen = set()
    for line in contents.splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.*)", line)
        if match is None or match[1] not in updates:
            lines.append(line)
            continue
        name = match[1]
        if name == "NB_CONVERSION_TIMEOUT_SECONDS":
            previous = match[2].split("#", 1)[0].strip().strip("'\"")
            if previous.isdigit():
                updates[name] = str(max(1800, int(previous)))
        if name not in seen:
            lines.append(f"{name}={updates[name]}")
            seen.add(name)
    lines.extend(f"{name}={value}" for name, value in updates.items() if name not in seen)
    return "\n".join(lines) + "\n"


def deploy(args):
    import pwd

    target = args.converter_root.resolve()
    seed = args.seed_root.resolve()
    env_file = args.env_file.resolve(strict=True)
    owner = pwd.getpwnam(args.user)
    if target == seed or target == Path(target.anchor):
        raise ValueError("Choose a separate converter root for this platform instance")
    env_contents = env_file.read_text(encoding="utf-8")
    # Read configuration before creating the target, so retries use the same settings.
    config_path = target / "converter" / "config.json"
    config_source = config_path if config_path.is_file() else seed / "converter" / "config.json"
    if not config_source.is_file():
        config_source = SOURCE / "config.example.json"
    config = json.loads(config_source.read_text(encoding="utf-8"))
    if not config_path.is_file():
        config.pop("sdk_workspace", None)
    image = config.get("sdk_image", "ubuntu-npu:v2.0.10.2")
    subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)
    # The API service grants Docker access through SupplementaryGroups=docker.
    subprocess.run(["sudo", "-u", args.user, "-g", "docker", "--", "docker", "image", "inspect", image],
                   check=True, stdout=subprocess.DEVNULL)

    target.mkdir(parents=True, exist_ok=True)
    converter = target / "converter"
    converter.mkdir(exist_ok=True)
    python = converter / ".venv" / "bin" / "python"
    if not python.is_file():
        creator = shutil.which("python3.11")
        if creator is None:
            creator = str(seed / "converter" / ".venv" / "bin" / "python")
        subprocess.run([creator, "-m", "venv", str(converter / ".venv")], check=True)
    # The seed package contains the original offline dependencies, and Git supplies Pillow.
    install = [str(python), "-m", "pip", "install", "--no-index",
               "--find-links", str(SOURCE / "wheelhouse")]
    if (seed / "wheelhouse").is_dir():
        install.extend(["--find-links", str(seed / "wheelhouse")])
    subprocess.run([*install, "-r", str(SOURCE / "requirements.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = target / "deployment-backups" / stamp
    backup.mkdir(parents=True, mode=0o700)
    installed = []
    destinations = [(SOURCE / name, converter / name) for name in MODULES]
    destinations.append((SOURCE / "convert.sh", target / "convert.sh"))
    for _, destination in destinations:
        if destination.exists():
            shutil.copy2(destination, backup / destination.name)
    shutil.copy2(env_file, backup / "platform.env")
    try:
        for source, destination in destinations:
            replace_file(source, destination)
            installed.append(destination)
        if not config_path.exists():
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            config_path.chmod(0o644)
        for name in ("tasks", "sdk-workspace"):
            directory = converter / name
            directory.mkdir(exist_ok=True)
            os.chown(directory, owner.pw_uid, owner.pw_gid)
            directory.chmod(0o750)
        # Import from the installed location using the actual API account.
        subprocess.run([
            "sudo", "-u", args.user, str(python), "-B", "-c",
            "import json, convert_nb_docker, nb_validation, point_quantization, PIL; "
            "assert point_quantization.RECIPE == 'point-pcint8-rgb-u8-v3'; "
            "assert PIL.__version__ == '11.0.0'; "
            "nb_validation.quality_limits(json.load(open('config.json')).get('validation_limits', {})); "
            "print('POINT_CONVERTER_V3_IMPORT_OK')",
        ], cwd=converter, check=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target, delete=False) as handle:
            staged_env = Path(handle.name)
            handle.write(update_environment(env_contents, target))
        try:
            replace_file(staged_env, env_file)
        finally:
            staged_env.unlink(missing_ok=True)
    except BaseException:
        for destination in reversed(installed):
            old_file = backup / destination.name
            if old_file.exists():
                replace_file(old_file, destination)
            else:
                destination.unlink(missing_ok=True)
        replace_file(backup / "platform.env", env_file)
        raise
    print(f"Converter installed at {target}")
    print(f"Platform environment updated at {env_file}")
    print(f"Deployment backup saved at {backup}")
    print("Restart this platform instance's API service to load the updated environment.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--converter-root", type=Path, default=Path("/opt/glh-converter-zys"))
    parser.add_argument("--seed-root", type=Path, default=Path("/opt/glh-converter"))
    parser.add_argument("--user", default="huaciq")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run this installer with sudo to write /opt and set the API account permissions")
    deploy(args)


if __name__ == "__main__":
    main()
