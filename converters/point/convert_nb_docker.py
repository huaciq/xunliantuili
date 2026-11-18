"""Native x86 SDK backend. Reuses the checked converter and quantization recipe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import convert_nb as converter


def main(*args):
    settings = json.loads((converter.ROOT / "config.json").read_text())
    image = settings.get("sdk_image", "ubuntu-npu:v2.0.10.2")
    # The shared converter builds ROOTFS/workspace/training/converter-jobs.
    # This root is a host staging folder, mounted into the SDK at /workspace.
    staging = Path(settings.get("sdk_workspace", str(converter.ROOT / "sdk-workspace"))).resolve()
    staging.mkdir(parents=True, exist_ok=True)

    def run_native(guest_work, export, *sdk_args):
        del export
        host_work = staging / guest_work.relative_to('/')
        legacy_work = converter.ROOT/'sdk-workspace'/guest_work.relative_to('/')
        if legacy_work != host_work:
            legacy_work.parent.mkdir(parents=True, exist_ok=True)
            if legacy_work.exists() or legacy_work.is_symlink():
                if legacy_work.resolve() != host_work:
                    raise FileExistsError('SDK workspace alias already points elsewhere')
            else:
                legacy_work.symlink_to(host_work, target_is_directory=True)
        uid, gid = os.getuid(), os.getgid()
        container_name = "glh-sdk-" + uuid.uuid4().hex
        identity = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        watcher = subprocess.Popen([sys.executable, str(converter.ROOT/'sdk_container_watchdog.py'),
                                    str(os.getpid()), identity, container_name],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        command = ["sudo", "-n", "docker", "run", "--rm", "--name", container_name,
                   "--label", "glh.training.sdk=true", "--network", "none",
                   *(["--cgroup-parent", settings["sdk_cgroup_parent"]]
                     if settings.get("sdk_cgroup_parent") else []),
                   "--memory", settings.get("sdk_memory", "7g"), "--memory-swap", settings.get("sdk_memory", "7g"),
                   "--cpus", str(settings.get("sdk_cpus", 2)),
                   "--mount", f"type=bind,source={staging / 'workspace'},target=/workspace",
                   "--workdir", str(guest_work),
                   "--env", "HOME=/root", "--env", "CUDA_VISIBLE_DEVICES=-1",
                   "--env", "OMP_NUM_THREADS=2", "--env", "MKL_NUM_THREADS=2",
                   "--env", "TF_NUM_INTRAOP_THREADS=2", "--env", "TF_NUM_INTEROP_THREADS=1",
                   "--env", "TF_CPP_MIN_LOG_LEVEL=2",
                   "--env", "VSIMULATOR_SHADER_CORE_COUNT=1",
                   "--env", "VSIMULATOR_CONFIG=" + converter.TARGET,
                   "--env", "PYTHONPATH=/root/acuity-toolkit-whl-6.30.22/bin",
                   "--entrypoint", "/bin/bash", image, "-c",
                   'python3 "$@"; result=$?; chown -R "' + str(uid) + ':' + str(gid) +
                   '" .; exit "$result"',
                   "sdk-stage", str(guest_work / "sdk_point_entry.py"),
                   "/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py", *sdk_args]
        print("SDK native stage:", sdk_args[0], flush=True)
        started = time.monotonic()
        wall_started = time.time()
        try:
            subprocess.run(command, check=True)
        finally:
            # Normal completion: the --rm container is already gone.
            # On owner SIGKILL the detached watchdog performs removal.
            watcher.terminate()
            watcher.wait()
            timing = host_work/'native_stage_times.json'
            stages = json.loads(timing.read_text()) if timing.exists() else []
            stages.append(dict(stage=sdk_args[0], seconds=time.monotonic()-started,
                               started_at=wall_started, finished_at=time.time(), container=container_name))
            timing.write_text(json.dumps(stages, indent=2))

    converter.main(*args, workspace_root=staging, sdk_runner=run_native,
                   validation_settings=settings.get("validation_limits"))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: convert_nb_docker.py INPUT.onnx DATASET OUTPUT.nb")
    main(*sys.argv[1:])
