"""Final-artifact gates, immutable publication and interrupted relay recovery."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import backup_models as backup


def make_run(root, run_id="test_run", machine="sub"):
    run = root / run_id
    run.mkdir(parents=True)
    prefix = "best_crop_ir_fixed"
    backup.write_json(run / "pipeline.json", {
        "run_id": run_id, "machine": machine, "backbone": "mobilenetv2",
        "prefix": prefix, "code_revision": "test-revision",
    })
    models = [f"{prefix}_{variant}.tflite" for variant in ("float", "int8", "npu_int8")]
    for model in models:
        (run / model).write_bytes(b"synthetic test artifact, not a model")
        (run / model.replace(".tflite", "_manifest.json")).write_text("{}")
    (run / f"{prefix}.keras").write_bytes(b"synthetic checkpoint")
    (run / "evaluation.csv").write_text("synthetic evaluation")
    backup.write_json(run / "evaluation.json", {
        "split": "validation", "artifacts": [{"model": str(run / model)} for model in models],
    })
    return run


def test_ready_requires_final_npu_model_and_evaluation(tmp_path):
    run = make_run(tmp_path)
    artifact = run / "best_crop_ir_fixed_npu_int8.tflite"
    contents = artifact.read_bytes()
    artifact.unlink()
    with pytest.raises(ValueError, match="Missing/empty final artifact"):
        backup.ready(argparse.Namespace(run_dir=run))
    assert not (run / "backup_ready.json").exists()
    artifact.write_bytes(contents)
    report = backup.read_json(run / "evaluation.json")
    report["artifacts"].pop()
    backup.write_json(run / "evaluation.json", report)
    with pytest.raises(ValueError, match="all three"):
        backup.ready(argparse.Namespace(run_dir=run))
    report["artifacts"].append({"model": str(artifact)})
    backup.write_json(run / "evaluation.json", report)
    backup.ready(argparse.Namespace(run_dir=run))
    backup.verify(run)


def test_publication_verifies_bytes_preserves_source_and_retries_ack(tmp_path, monkeypatch):
    run = make_run(tmp_path / "source", machine="mac")
    backup.ready(argparse.Namespace(run_dir=run))
    config = tmp_path / "config.json"
    target = tmp_path / "Shared backup"
    backup.write_json(config, {"destination_root": str(target)})
    args = argparse.Namespace(config=config, run_dir=run)
    acknowledge = backup.acknowledge
    monkeypatch.setattr(backup, "acknowledge", lambda *a: (_ for _ in ()).throw(OSError("interrupted ack")))
    with pytest.raises(OSError, match="interrupted ack"):
        backup.publish(args)
    assert not (run / "backup_done.json").exists()
    destination = target / "keras/mobilenetv2/mac/test_run"
    assert backup.verify(destination) == backup.verify(run)
    monkeypatch.setattr(backup, "acknowledge", acknowledge)
    backup.publish(args)
    assert backup.completed(run)
    assert (run / "best_crop_ir_fixed.keras").is_file()
    backup.publish(args)
    assert len(list((target / "keras/mobilenetv2/mac").iterdir())) == 1


def test_changed_source_or_conflicting_destination_never_acknowledged(tmp_path):
    run = make_run(tmp_path / "source", machine="mac")
    backup.ready(argparse.Namespace(run_dir=run))
    config = tmp_path / "config.json"
    target = tmp_path / "backup"
    backup.write_json(config, {"destination_root": str(target)})
    args = argparse.Namespace(config=config, run_dir=run)
    artifact = run / "best_crop_ir_fixed_float.tflite"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        backup.publish(args)
    assert not (run / "backup_done.json").exists()
    (run / "backup_ready.json").unlink()
    backup.ready(argparse.Namespace(run_dir=run))
    backup.publish(args)
    (run / "backup_done.json").unlink()
    (run / "backup_ready.json").unlink()
    artifact.write_bytes(b"conflicting run")
    backup.ready(argparse.Namespace(run_dir=run))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        backup.publish(args)
    assert not (run / "backup_done.json").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../../escape"])
def test_manifest_paths_cannot_escape(tmp_path, name):
    run = make_run(tmp_path)
    backup.ready(argparse.Namespace(run_dir=run))
    marker = run / "backup_ready.json"
    data = backup.read_json(marker)
    data["files"][name] = {"size": 1, "sha256": "0" * 64}
    backup.write_json(marker, data)
    with pytest.raises(ValueError, match="Invalid artifact path"):
        backup.verify(run)


def test_relay_retries_after_destination_failure(tmp_path, monkeypatch):
    run = make_run(tmp_path / "source with spaces")
    backup.ready(argparse.Namespace(run_dir=run))
    destination = tmp_path / "Shared backup"
    config = tmp_path / "config.json"
    backup.write_json(config, {
        "cache_root": str(tmp_path / "cache"),
        "source": {"host": "sub", "root": str(run.parent), "helper": "helper"},
        "target": {"host": "mac", "root": str(destination), "helper": "helper"},
    })
    available = False

    def remote(host, helper, action, *argv):
        options = dict(zip(argv[::2], map(str, argv[1::2])))
        if action == "pending":
            return json.dumps([] if backup.completed(run) else [{"run_dir": str(run), "ready_sha256": backup.digest(run / "backup_ready.json")}])
        if action == "stage":
            if not available:
                raise OSError("mac offline")
            Path(options["--path"]).mkdir(parents=True, exist_ok=True)
            return ""
        if action == "commit":
            return backup.commit(options["--staging"], options["--destination-root"])
        if action == "ack":
            backup.acknowledge(run, options["--fingerprint"], options["--destination"])
            return ""
        raise AssertionError(action)

    def rsync(command, **kwargs):
        src, dst = command[-2:]
        import shlex
        # Modern rsync protects remote args itself; explicit shell quotes then
        # become literal path characters unless legacy argument handling is used.
        def path(value, host):
            if value.startswith(host + ":"):
                value = value[len(host) + 1:]
                if "--old-args" in command:
                    value = shlex.split(value)[0]
            return value
        shutil.copytree(path(src, "sub"), path(dst, "mac"), dirs_exist_ok=True)

    monkeypatch.setattr(backup, "remote", remote)
    monkeypatch.setattr(backup.subprocess, "run", rsync)
    args = argparse.Namespace(config=config)
    with pytest.raises(RuntimeError, match="remain pending"):
        backup.relay(args)
    assert not backup.completed(run)
    available = True
    backup.relay(args)
    assert backup.completed(run)
    backup.relay(args)
    assert backup.verify(destination / "keras/mobilenetv2/sub/test_run") == backup.verify(run)


@pytest.mark.parametrize("fail_stage", ["validate", "train", "convert", "evaluate", ""])
@pytest.mark.parametrize("explicit_run_id", [True, False, None])
def test_pipeline_only_queues_after_every_stage_passes(tmp_path, fail_stage, explicit_run_id, monkeypatch):
    import os

    project = tmp_path / "project"
    (project / "scripts/keras").mkdir(parents=True)
    (project / ".venv-tf/bin").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-qm", "fixture"], check=True)
    config = tmp_path / "config.json"
    backup.write_json(config, {"machine": "sub"})
    stub = tmp_path / "stage.py"
    stub.write_text('''import json,os,sys
from pathlib import Path
stage=sys.argv[1]
args=sys.argv[2:]
if stage=="-m":
 stage="validate" if args[0]=="common.validate_fixed_splits" else "evaluate"
 args=args[1:]
if stage==os.environ["TEST_FAIL_STAGE"]: sys.exit(7)
def arg(name): return args[args.index(name)+1]
if stage in {"train","convert"}:
 root=Path(arg("--output-dir")); prefix="best_model_fixed" if arg("--model-type")=="dual" else "best_crop_ir_fixed"
 if stage=="train": (root/(prefix+".keras")).write_bytes(b"fixture")
 else:
  for variant in ["float","int8","npu_int8"]:
   (root/(prefix+"_"+variant+".tflite")).write_bytes(b"fixture")
   (root/(prefix+"_"+variant+"_manifest.json")).write_text("{}")
if stage=="evaluate":
 models=args[args.index("--models")+1:]
 Path(arg("--report-json")).write_text(json.dumps({"split":"validation","artifacts":[{"model":p} for p in models]}))
 Path(arg("--report-csv")).write_text("fixture")
''')
    for path, stage in [(project / ".venv-tf/bin/python", ""),
                        (project / "scripts/keras/run_keras_train.sh", "train"),
                        (project / "scripts/keras/run_keras_convert.sh", "convert")]:
        path.write_text(f'#!/bin/sh\nexec "$TEST_PYTHON" "$TEST_STUB" {stage} "$@"\n')
        path.chmod(0o755)
    wrapper = Path(backup.__file__).parent / "keras/run_fixed_split.sh"
    env = {**os.environ, "MODEL_PROJECT_ROOT": str(project), "MODEL_BACKUP_SCRIPT": backup.__file__,
           "MODEL_BACKUP_CONFIG": str(config), "TEST_STUB": str(stub), "TEST_PYTHON": sys.executable,
           "TEST_FAIL_STAGE": fail_stage}
    options = [] if explicit_run_id is None else ["--model-type", "crop_ir"]
    if explicit_run_id:
        options += ["--run-id", "fixture"]
    command = ["bash", str(wrapper)]
    if not fail_stage and explicit_run_id is None:
        home = tmp_path / "installed-home"
        monkeypatch.setenv("HOME", str(home))
        backup.install(argparse.Namespace(machine="sub", cron=False, destination_root=None,
                                         project_root=project, config=config))
        env["HOME"] = str(home)
        command = [str(home / ".local/bin/run-model-pipeline")]
    result = subprocess.run([*command, *options],
                            env=env, text=True, capture_output=True)
    runs = list((project / "model/keras").iterdir())
    assert len(runs) == 1
    marker = runs[0] / "backup_ready.json"
    assert result.returncode == (7 if fail_stage else 0), result.stderr + result.stdout
    assert marker.exists() == (not fail_stage)
    assert not (marker.parent / "backup_done.json").exists()
    if marker.exists():
        snapshot = backup.verify(marker.parent)
        if explicit_run_id:
            repeat = subprocess.run([*command, *options, "--force"], env=env, text=True, capture_output=True)
            assert repeat.returncode != 0
            assert backup.verify(marker.parent) == snapshot


def test_install_cron_preserves_other_jobs_and_replaces_only_own_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    previous = "17 * * * * existing-job\n*/5 * * * * obsolete # access-liveness-model-backup\n"
    installed = []

    def remote_config(command, **kwargs):
        machine = "sub" if command[-2] == "sub" else "mac"
        return json.dumps({"home": "/remote", "config": {"machine": machine,
            "project_root": "/remote/project", "destination_root": "/Shared backup", "helper": "/remote/helper.py"}})

    def crontab(command, **kwargs):
        if command == ["crontab", "-l"]:
            return subprocess.CompletedProcess(command, 0, previous, "")
        assert command == ["crontab", "-"]
        installed.append(kwargs["input"])

    monkeypatch.setattr(backup.subprocess, "check_output", remote_config)
    monkeypatch.setattr(backup.subprocess, "run", crontab)
    args = argparse.Namespace(machine="company", cron=True, project_root=tmp_path,
                              source_host="sub", target_host="mac", destination_root=None,
                              config=tmp_path / "config.json")
    backup.install(args)
    assert installed[0].count("# access-liveness-model-backup") == 1
    assert "17 * * * * existing-job\n" in installed[0]
    assert "obsolete" not in installed[0]
    assert "*/5 * * * * /usr/bin/python3" in installed[0]
    settings = backup.read_json(args.config)
    assert settings["source"]["root"] == "/remote/project/model"
    assert Path(settings["helper"]).read_bytes() == Path(backup.__file__).read_bytes()


def test_corrupt_done_marker_is_requeued_without_blocking_other_runs(tmp_path, capsys):
    run = make_run(tmp_path, "corrupt_ack")
    other = make_run(tmp_path, "other_run")
    for item in [run, other]:
        backup.ready(argparse.Namespace(run_dir=item))
    capsys.readouterr()
    (run / "backup_done.json").write_text("interrupted acknowledgment")
    backup.pending(argparse.Namespace(root=tmp_path))
    entries = json.loads(capsys.readouterr().out)
    assert {row["run_dir"] for row in entries} == {str(run), str(other)}


def test_concurrent_relay_skips_before_contacting_hosts(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    backup.write_json(config, {"cache_root": str(tmp_path)})

    def unexpected_remote(*args):
        raise AssertionError("A second relay must not contact any host")

    monkeypatch.setattr(backup, "remote", unexpected_remote)
    with (tmp_path / ".relay.lock").open("a") as lock:
        backup.fcntl.flock(lock, backup.fcntl.LOCK_EX | backup.fcntl.LOCK_NB)
        backup.relay(argparse.Namespace(config=config))
    assert "Another relay is running; skipping" in capsys.readouterr().out
