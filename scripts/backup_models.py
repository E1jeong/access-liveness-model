"""Verified run backup: local publication on mac, SSH relay from company WSL."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


DEFAULT_CONFIG = Path.home() / ".config/access-liveness-model/backup.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
CONTROL = {"backup_ready.json", "backup_done.json"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


SAFE_FOLDER_PATTERN = re.compile(r"^[A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ _+.,\(\)\[\]\-]+$")


def safe_folder_name(value):
    if not isinstance(value, str):
        raise ValueError(f"Folder name must be a string: {value!r}")
    value = value.strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid folder name: {value!r}")
    if "/" in value or "\\" in value or "\0" in value:
        raise ValueError(f"Folder name contains path separators: {value!r}")
    if not SAFE_FOLDER_PATTERN.fullmatch(value):
        raise ValueError(f"Folder name contains forbidden characters: {value!r}")
    if len(value.encode("utf-8")) > 255:
        raise ValueError(f"Folder name exceeds 255 bytes: {value!r}")
    return value


def resolve_backbone_dir(keras_root, backbone):
    backbone = component(backbone)
    if not keras_root.is_dir():
        return backbone
    lower_backbone = backbone.lower()
    for entry in keras_root.iterdir():
        if entry.is_dir() and entry.name.lower() == lower_backbone:
            return entry.name
    return backbone


def determine_backup_folder_name(metadata):
    if metadata.get("backup_name"):
        return safe_folder_name(metadata["backup_name"])

    run_id = metadata.get("run_id", "")
    match = re.match(r"^(\d{8})", run_id)
    if match:
        date_str = match.group(1)
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    model_type = metadata.get("model_type", "dual")
    if model_type == "dual":
        model_tag = "2input"
    elif model_type == "crop_ir":
        model_tag = "1input"
    elif model_type == "crop_rgb":
        model_tag = "1input(rgb)"
    else:
        model_tag = model_type

    parts = [f"{date_str}_{model_tag} 12클래스"]

    note = (metadata.get("note") or "").strip()
    if note:
        parts.append(note)

    command = metadata.get("command") or []
    if isinstance(command, str):
        command = shlex.split(command)

    def has_flag(flag):
        return any(arg == flag or arg.startswith(flag + "=") for arg in command)

    def get_arg_val(flag):
        for i, arg in enumerate(command):
            if arg == flag and i + 1 < len(command):
                return command[i + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
        return None

    depth = has_flag("--aux-depth")
    residual = has_flag("--aux-residual")
    binary_pad = has_flag("--aux-binary-pad")
    supcon = has_flag("--aux-supcon")

    if depth and residual and binary_pad and supcon:
        parts.append("보조헤드 4개")
    else:
        active_heads = []
        if depth:
            active_heads.append("depth")
        if residual:
            active_heads.append("잔차")
        if binary_pad:
            active_heads.append("이진pad")
        if supcon:
            active_heads.append("supcon")
        if active_heads:
            parts.append(f"보조헤드({', '.join(active_heads)})")

    loss = get_arg_val("--loss") or get_arg_val("--loss-type")
    if loss and loss.lower() == "focal":
        parts.append("focal")

    if has_flag("--augment-display-artifacts"):
        parts.append("모아레on")

    freeze = get_arg_val("--freeze-backbone-epochs")
    if freeze and freeze not in {"0", "0.0"}:
        parts.append(f"동결{freeze}에폭")

    machine = metadata.get("machine", "unknown")
    if machine == "mac" or has_flag("--cpu") or has_flag("--device=cpu") or "cpu" in run_id.lower():
        parts.append("cpu학습")
    else:
        parts.append("gpu학습")

    epochs = get_arg_val("--epochs") or "10"
    epoch_tag = f"({epochs}에폭)"

    body = " + ".join(parts)
    folder_name = f"{body} {epoch_tag} - {machine}"
    return safe_folder_name(folder_name)


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid backup path component: {value!r}")
    return value


def file_path(root, name):
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or any(p in {".", ".."} for p in relative.parts):
        raise ValueError(f"Invalid artifact path: {name}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes run directory: {name}")
    return path


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not backup artifacts: {path}")
        if path.is_file() and path.name not in CONTROL:
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size, "sha256": digest(path),
            }
    return result


def verify(root):
    root = Path(root)
    marker = root / "backup_ready.json"
    if marker.is_symlink():
        raise ValueError("Ready marker cannot be a symlink")
    manifest = read_json(marker)
    if manifest.get("version") != 1 or not manifest.get("files"):
        raise ValueError("Invalid or empty backup manifest")
    for key in ("machine", "backbone", "run_id"):
        component(manifest[key])
    if "backup_folder" in manifest:
        safe_folder_name(manifest["backup_folder"])
    for name in manifest["files"]:
        file_path(root, name)
    if inventory(root) != manifest["files"]:
        raise ValueError(f"Artifact inventory/hash mismatch: {root}")
    return manifest, digest(marker)


def start(args):
    config = read_json(args.config)
    machine = component(config["machine"])
    if machine != "sub" and not config.get("destination_root"):
        raise ValueError("Local backup requires destination_root; sub queues for the WSL relay")
    run_id = component(args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + args.model_type)
    root = Path(args.output_root).resolve()
    if not root.is_relative_to((Path.cwd() / "model").resolve()):
        raise ValueError("Run output must be under the project's model/ directory")
    run = root / run_id
    run.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    pipeline_data = {
        "run_id": run_id, "machine": machine, "backbone": component(args.backbone),
        "model_type": args.model_type, "prefix": args.prefix,
        "code_revision": revision, "worktree_dirty": dirty, "command": args.command,
    }
    if getattr(args, "note", None):
        pipeline_data["note"] = args.note
    if getattr(args, "backup_name", None):
        pipeline_data["backup_name"] = safe_folder_name(args.backup_name)
    write_json(run / "pipeline.json", pipeline_data)
    print(run)


def ready(args):
    run = Path(args.run_dir)
    if any((run / name).exists() for name in CONTROL):
        raise FileExistsError("Run already registered for backup")
    metadata = read_json(run / "pipeline.json")
    prefix = component(metadata["prefix"])
    models = {f"{prefix}_{variant}.tflite" for variant in ("float", "int8", "npu_int8")}
    required = models | {f"{prefix}.keras", "evaluation.json", "evaluation.csv"}
    required |= {name.replace(".tflite", "_manifest.json") for name in models}
    for name in required:
        path = file_path(run, name)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty final artifact: {path}")
    report = read_json(run / "evaluation.json")
    if report.get("split") != "validation" or {Path(row["model"]).name for row in report.get("artifacts", [])} != models:
        raise ValueError("Evaluation must cover all three final TFLite artifacts on validation")
    write_json(run / "backup_ready.json", {
        "version": 1, "run_id": metadata["run_id"], "machine": metadata["machine"],
        "backbone": metadata["backbone"], "code_revision": metadata["code_revision"],
        "backup_folder": determine_backup_folder_name(metadata),
        "files": inventory(run),
    })
    print(f"Backup ready: {run}")


def commit(staging, destination_root):
    staging, destination_root = Path(staging).resolve(), Path(destination_root).resolve()
    if staging.parent != destination_root / ".incoming":
        raise ValueError("Staging must be directly under destination_root/.incoming")
    destination_root.mkdir(parents=True, exist_ok=True)
    with (destination_root / ".backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, fingerprint = verify(staging)
        backbone_dir = resolve_backbone_dir(destination_root / "keras", manifest["backbone"])
        folder_name = safe_folder_name(manifest.get("backup_folder") or manifest["run_id"])
        destination = destination_root / "keras" / backbone_dir / folder_name
        if destination.exists():
            existing_manifest, existing = verify(destination)
            if existing == fingerprint:
                shutil.rmtree(staging)
                return str(destination)
            if existing_manifest.get("run_id") == manifest["run_id"]:
                raise FileExistsError(f"Conflicting backup for run {manifest['run_id']}; refusing overwrite: {destination}")
            collision_name = safe_folder_name(f"{folder_name}_{manifest['run_id']}")
            destination = destination_root / "keras" / backbone_dir / collision_name
            if destination.exists():
                _, existing_collision = verify(destination)
                if existing_collision == fingerprint:
                    shutil.rmtree(staging)
                    return str(destination)
                raise FileExistsError(f"Conflicting backup; refusing overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return str(destination)


def acknowledge(run, fingerprint, destination):
    _, current = verify(Path(run))
    if current != fingerprint:
        raise ValueError("Source manifest changed before acknowledgment")
    write_json(Path(run) / "backup_done.json", {
        "ready_sha256": fingerprint, "destination": destination,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    })


def completed(run):
    marker = run / "backup_done.json"
    try:
        return marker.is_file() and read_json(marker).get("ready_sha256") == digest(run / "backup_ready.json")
    except (OSError, ValueError):
        # A damaged acknowledgment must not stall unrelated queued runs.
        return False


def publish(args):
    config = read_json(args.config)
    run = Path(args.run_dir).resolve()
    manifest, fingerprint = verify(run)
    if completed(run):
        print(f"Already backed up: {run}")
        return
    destination_root = Path(config["destination_root"])
    staging = destination_root / ".incoming" / (manifest["machine"] + "_" + fingerprint)
    staging.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary directory prevents simultaneous copies sharing mutable staging.
    staging = Path(tempfile.mkdtemp(prefix=staging.name + "_", dir=staging.parent))
    shutil.copytree(run, staging, dirs_exist_ok=True, ignore=shutil.ignore_patterns("backup_done.json"))
    destination = commit(staging, destination_root)
    acknowledge(run, fingerprint, destination)
    print(f"Backup verified: {destination}")


def pending(args):
    result = []
    root = Path(args.root).resolve()
    for marker in sorted(root.rglob("backup_ready.json")):
        if marker.is_symlink() or not marker.resolve().is_relative_to(root):
            raise ValueError("Ready marker escapes source root")
        if not completed(marker.parent):
            result.append({"run_dir": str(marker.parent), "ready_sha256": digest(marker)})
    print(json.dumps(result))


def remote(host, helper, *args):
    component(host)
    return subprocess.check_output(SSH + [host, shlex.join(["python3", helper, *map(str, args)])], text=True)


def relay(args):
    config = read_json(args.config)
    cache = Path(config["cache_root"]).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".relay.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another relay is running; skipping")
            return
        source, target = config["source"], config["target"]
        entries = json.loads(remote(source["host"], source["helper"], "pending", "--root", source["root"]))
        failures = []
        for entry in entries:
            try:
                fingerprint = entry["ready_sha256"]
                if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
                    raise ValueError("Invalid ready fingerprint")
                source_run = PurePosixPath(entry["run_dir"])
                if not source_run.is_absolute() or ".." in source_run.parts or not source_run.is_relative_to(PurePosixPath(source["root"])):
                    raise ValueError("Run escapes configured source root")
                local = cache / fingerprint
                local.mkdir(exist_ok=True)
                transport = shlex.join(SSH)
                # Paths are shell-quoted explicitly for mac's protocol-29 rsync.
                subprocess.run(["rsync", "--old-args", "-rltc", "--exclude=backup_done.json", "-e", transport, "--",
                                source["host"] + ":" + shlex.quote(str(source_run) + "/"), str(local) + "/"], check=True)
                manifest, actual = verify(local)
                if actual != fingerprint:
                    raise ValueError("Transferred marker differs from queued marker")
                staging = PurePosixPath(target["root"]) / ".incoming" / (manifest["machine"] + "_" + fingerprint)
                remote(target["host"], target["helper"], "stage", "--path", staging, "--destination-root", target["root"])
                subprocess.run(["rsync", "--old-args", "-rltc", "-e", transport, "--", str(local) + "/",
                                target["host"] + ":" + shlex.quote(str(staging) + "/")], check=True)
                destination = remote(target["host"], target["helper"], "commit", "--staging", staging,
                                     "--destination-root", target["root"]).strip()
                remote(source["host"], source["helper"], "ack", "--run-dir", source_run,
                       "--fingerprint", fingerprint, "--destination", destination)
                shutil.rmtree(local)
                print(f"Backup verified: {source['host']}:{source_run} -> {destination}", flush=True)
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                failures.append(str(error))
                print(f"Backup pending; next relay will retry: {entry['run_dir']}: {error}", file=sys.stderr, flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} backup(s) remain pending")


def install(args):
    """Install immutable helper/wrapper copies without changing host checkouts."""
    if args.cron and args.machine != "company":
        raise ValueError("The relay cron belongs on company WSL")
    if args.machine == "mac" and (not args.destination_root or not Path(args.destination_root).is_dir()):
        raise ValueError("mac installation requires an existing --destination-root")
    helper = Path(__file__).read_bytes()
    wrapper = Path(__file__).parent / "keras/run_fixed_split.sh"
    wrapper_bytes = wrapper.read_bytes()
    version = hashlib.sha256(helper + wrapper_bytes).hexdigest()[:16]
    base = Path.home() / ".local/share/access-liveness-model/backup" / version
    base.mkdir(parents=True, exist_ok=True)
    (base / "backup_models.py").write_bytes(helper)
    (base / "run_fixed_split.sh").write_bytes(wrapper_bytes)
    settings = {"machine": args.machine, "project_root": str(Path(args.project_root).resolve())}
    if args.machine == "mac":
        settings["destination_root"] = str(Path(args.destination_root).resolve())
    if args.machine == "company":
        settings["cache_root"] = str(Path.home() / ".cache/access-liveness-model/relay")
        for key, host in (("source", args.source_host), ("target", args.target_host)):
            component(host)
            probe = "import json; from pathlib import Path; p=Path.home()/'.config/access-liveness-model/backup.json'; print(json.dumps({'home':str(Path.home()),'config':json.loads(p.read_text())}))"
            info = json.loads(subprocess.check_output(SSH + [host, shlex.join(["python3", "-c", probe])], text=True))
            remote_config = info["config"]
            if remote_config["machine"] != ("sub" if key == "source" else "mac"):
                raise ValueError(f"Unexpected machine role on {host}")
            root = str(Path(remote_config["project_root"]) / "model") if key == "source" else remote_config["destination_root"]
            settings[key] = {"host": host, "root": root, "helper": remote_config["helper"]}
    settings["helper"] = str(base / "backup_models.py")
    config = Path(args.config).expanduser().resolve()
    config.parent.mkdir(parents=True, exist_ok=True)
    write_json(config, settings)
    config.chmod(0o600)
    if args.machine != "company":
        launcher = Path.home() / ".local/bin/run-model-pipeline"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "export MODEL_PROJECT_ROOT=" + shlex.quote(settings["project_root"]) + "\n"
            + "export MODEL_BACKUP_SCRIPT=" + shlex.quote(str(base / "backup_models.py")) + "\n"
            + "export MODEL_BACKUP_CONFIG=" + shlex.quote(str(config)) + "\n"
            + "exec bash " + shlex.quote(str(base / "run_fixed_split.sh")) + ' "$@"\n'
        )
        launcher.chmod(0o755)
    if args.cron:
        state = Path.home() / ".local/state/access-liveness-model"
        state.mkdir(parents=True, exist_ok=True)
        previous = subprocess.run(["crontab", "-l"], text=True, capture_output=True)
        if previous.returncode and "no crontab" not in previous.stderr:
            raise RuntimeError(previous.stderr)
        tag = "# access-liveness-model-backup"
        lines = [line for line in previous.stdout.splitlines() if not line.endswith(tag)]
        command = shlex.join(["/usr/bin/python3", str(base / "backup_models.py"), "relay", "--config", str(config)])
        lines.append("*/5 * * * * " + command + " >> " + shlex.quote(str(state / "relay.log")) + " 2>&1 " + tag)
        subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
    print(json.dumps({"version": version, "config": str(config), "helper": str(base / "backup_models.py")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("start")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", required=True)
    p.add_argument("--run-id")
    p.add_argument("--model-type", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--note")
    p.add_argument("--backup-name")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = commands.add_parser("ready")
    p.add_argument("--run-dir", required=True)
    p = commands.add_parser("publish")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run-dir", required=True)
    p = commands.add_parser("pending")
    p.add_argument("--root", required=True)
    p = commands.add_parser("relay")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = commands.add_parser("install")
    p.add_argument("--machine", required=True, choices=["sub", "mac", "company"])
    p.add_argument("--project-root", default=Path.cwd())
    p.add_argument("--destination-root")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--source-host", default="sub")
    p.add_argument("--target-host", default="mac")
    p.add_argument("--cron", action="store_true")
    p = commands.add_parser("stage")
    p.add_argument("--path", required=True)
    p.add_argument("--destination-root", required=True)
    p = commands.add_parser("commit")
    p.add_argument("--staging", required=True)
    p.add_argument("--destination-root", required=True)
    p = commands.add_parser("ack")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.action == "commit":
        print(commit(args.staging, args.destination_root))
    elif args.action == "ack":
        acknowledge(args.run_dir, args.fingerprint, args.destination)
    elif args.action == "stage":
        path = Path(args.path).resolve()
        if path.parent != Path(args.destination_root).resolve() / ".incoming":
            raise ValueError("Staging path must be directly under destination_root/.incoming")
        path.mkdir(parents=True, exist_ok=True)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Backup error: {error}")
