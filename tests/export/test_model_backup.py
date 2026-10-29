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
    manifest, _ = backup.verify(run)
    folder_name = manifest.get("backup_folder") or manifest["run_id"]
    destination = target / "keras/mobilenetv2" / folder_name
    assert backup.verify(destination) == backup.verify(run)
    monkeypatch.setattr(backup, "acknowledge", acknowledge)
    backup.publish(args)
    assert backup.completed(run)
    assert (run / "best_crop_ir_fixed.keras").is_file()
    backup.publish(args)
    assert len(list((target / "keras/mobilenetv2").iterdir())) == 1


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
    manifest, _ = backup.verify(run)
    folder_name = manifest.get("backup_folder") or manifest["run_id"]
    assert backup.verify(destination / "keras/mobilenetv2" / folder_name) == backup.verify(run)


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
 if stage=="train":
  assert arg("--weights")==os.environ["TEST_WEIGHTS"]
  assert not {"--conv1-reduction","--no-gray-imagenet-init","--rgb-weights"}.intersection(args)
  (root/(prefix+".keras")).write_bytes(b"fixture")
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
           "TEST_FAIL_STAGE": fail_stage, "TEST_WEIGHTS": "none" if explicit_run_id else "imagenet"}
    options = [] if explicit_run_id is None else ["--model-type", "crop_ir"]
    if explicit_run_id:
        options += ["--run-id", "fixture", "--weights", "none"]
    command = ["bash", str(wrapper)]
    if not fail_stage and explicit_run_id is None:
        home = tmp_path / "installed-home"
        monkeypatch.setenv("HOME", str(home))
        shutil.copy2(wrapper, project / "scripts/keras/run_fixed_split.sh")
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


@pytest.mark.parametrize("machine", ["sub", "mac"])
def test_installed_launcher_uses_updated_checkout_wrapper(tmp_path, monkeypatch, machine):
    monkeypatch.setenv("HOME", str(tmp_path / "installed home"))
    project = tmp_path / "project checkout"
    wrapper = project / "scripts/keras/run_fixed_split.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('#!/bin/bash\nprintf "old checkout\\n"\n')
    destination = tmp_path / "Shared backup"
    destination.mkdir()
    config = tmp_path / "config.json"
    backup.install(argparse.Namespace(
        machine=machine, cron=False, destination_root=destination,
        project_root=project, config=config,
    ))
    settings = backup.read_json(config)
    installed_wrapper = Path(settings["helper"]).with_name("run_fixed_split.sh")
    installed_before = installed_wrapper.read_bytes()
    wrapper.write_text('#!/bin/bash\nprintf "checkout:%s\\n" "$@"\n')
    launcher = Path.home() / ".local/bin/run-model-pipeline"
    result = subprocess.run([str(launcher), "--weights", "none", "argument with spaces"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "checkout:--weights", "checkout:none", "checkout:argument with spaces",
    ]
    assert installed_wrapper.read_bytes() == installed_before


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


def test_safe_folder_name_validation():
    valid_names = [
        "20261027_2input 12클래스 + 보조헤드 4개 + cpu학습 (40에폭) - mac",
        "20261027_1input 12클래스 + 야외이미지(곡면포함) + gpu학습 (10에폭) - sub",
        "custom_folder-name.1",
        "폴더명 [v2] (테스트)",
    ]
    for name in valid_names:
        assert backup.safe_folder_name(name) == name

    invalid_names = [
        "",
        "   ",
        ".",
        "..",
        "nested/path",
        "has\\backslash",
        "../escape",
        "forbidden*char",
        "pipe|name",
        "colon:name",
    ]
    for name in invalid_names:
        with pytest.raises(ValueError):
            backup.safe_folder_name(name)


def test_determine_backup_folder_name_variations():
    # 1. Custom backup_name override
    meta_custom = {
        "run_id": "20261027_dual",
        "backup_name": "내가_직접_지정한_이름",
    }
    assert backup.determine_backup_folder_name(meta_custom) == "내가_직접_지정한_이름"

    # 2. Dual with 4 auxiliary heads, cpu training on mac, 40 epochs, without note
    meta_dual_4aux = {
        "run_id": "20261027T015602Z_dual_preset_f_cpu40",
        "model_type": "dual",
        "machine": "mac",
        "command": [
            "run_fixed_split.sh", "--epochs", "40",
            "--aux-depth", "--aux-residual", "--aux-binary-pad", "--aux-supcon",
        ],
    }
    name = backup.determine_backup_folder_name(meta_dual_4aux)
    assert name == "20261027_2input 12클래스 + 보조헤드 4개 + cpu학습 (40에폭) - mac"
    # Ensure default does NOT include outdoor note
    assert "야외" not in name

    # 3. With explicit note
    meta_with_note = {
        "run_id": "20261027T015602Z_dual_preset_f_cpu40",
        "model_type": "dual",
        "machine": "mac",
        "note": "야외이미지(곡면포함)",
        "command": [
            "run_fixed_split.sh", "--epochs", "40",
            "--aux-depth", "--aux-residual", "--aux-binary-pad", "--aux-supcon",
        ],
    }
    name_with_note = backup.determine_backup_folder_name(meta_with_note)
    assert name_with_note == "20261027_2input 12클래스 + 야외이미지(곡면포함) + 보조헤드 4개 + cpu학습 (40에폭) - mac"

    # 4. Partial auxiliary heads + focal loss + sub GPU
    meta_sub = {
        "run_id": "20261028_crop_ir",
        "model_type": "crop_ir",
        "machine": "sub",
        "command": [
            "run_fixed_split.sh", "--epochs", "20",
            "--aux-depth", "--aux-residual",
            "--loss-type", "focal",
        ],
    }
    name_sub = backup.determine_backup_folder_name(meta_sub)
    assert name_sub == "20261028_1input 12클래스 + 보조헤드(depth, 잔차) + focal + gpu학습 (20에폭) - sub"


def test_case_insensitive_backbone_matching(tmp_path):
    target = tmp_path / "target"
    # Pre-existing camelCase directory
    existing_dir = target / "keras/mobileNetV2"
    existing_dir.mkdir(parents=True)

    run = make_run(tmp_path / "source", machine="mac")
    backup.ready(argparse.Namespace(run_dir=run))
    config = tmp_path / "config.json"
    backup.write_json(config, {"destination_root": str(target)})

    backup.publish(argparse.Namespace(config=config, run_dir=run))
    assert backup.completed(run)
    # Target should be placed directly inside mobileNetV2 (not creating a second mobilenetv2)
    keras_dirs = [d.name for d in (target / "keras").iterdir()]
    assert keras_dirs == ["mobileNetV2"]
    manifest, _ = backup.verify(run)
    folder_name = manifest.get("backup_folder") or manifest["run_id"]
    assert (existing_dir / folder_name).is_dir()


def test_collision_with_different_run_id_uses_disambiguation_suffix(tmp_path):
    target = tmp_path / "target"
    config = tmp_path / "config.json"
    backup.write_json(config, {"destination_root": str(target)})

    # First run
    run1 = make_run(tmp_path / "run1", run_id="run_alpha", machine="mac")
    # Same custom folder name for both
    p1 = backup.read_json(run1 / "pipeline.json")
    p1["backup_name"] = "공통_백업_폴더"
    backup.write_json(run1 / "pipeline.json", p1)
    backup.ready(argparse.Namespace(run_dir=run1))
    backup.publish(argparse.Namespace(config=config, run_dir=run1))
    assert backup.completed(run1)

    # Second run with different run_id and different contents
    run2 = make_run(tmp_path / "run2", run_id="run_beta", machine="mac")
    (run2 / "best_crop_ir_fixed_float.tflite").write_bytes(b"different content for beta")
    p2 = backup.read_json(run2 / "pipeline.json")
    p2["backup_name"] = "공통_백업_폴더"
    backup.write_json(run2 / "pipeline.json", p2)
    backup.ready(argparse.Namespace(run_dir=run2))
    backup.publish(argparse.Namespace(config=config, run_dir=run2))
    assert backup.completed(run2)

    # Check both exist under keras/mobilenetv2
    dest1 = target / "keras/mobilenetv2/공통_백업_폴더"
    dest2 = target / "keras/mobilenetv2/공통_백업_폴더_run_beta"
    assert backup.verify(dest1) == backup.verify(run1)
    assert backup.verify(dest2) == backup.verify(run2)
