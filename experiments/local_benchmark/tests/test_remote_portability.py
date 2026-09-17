import json
from pathlib import Path

from experiments.local_benchmark.adapters import hermes, openclaw
from experiments.local_benchmark.evaluator import summarize


def test_hermes_runtime_honors_environment(monkeypatch, tmp_path: Path) -> None:
    python_path = tmp_path / "hermes-python"
    cli_path = tmp_path / "hermes"
    python_path.touch()
    cli_path.touch()
    monkeypatch.setenv("LOCAL_BENCHMARK_HERMES_PYTHON", str(python_path))
    monkeypatch.setenv("LOCAL_BENCHMARK_HERMES_CLI", str(cli_path))

    assert hermes._hermes_runtime() == (cli_path, python_path)


def test_openclaw_runtime_honors_environment(monkeypatch, tmp_path: Path) -> None:
    node_bin = tmp_path / "bin"
    cli_path = tmp_path / "openclaw"
    node_bin.mkdir()
    cli_path.touch()
    monkeypatch.setenv("LOCAL_BENCHMARK_NODE_BIN_DIR", str(node_bin))
    monkeypatch.setenv("LOCAL_BENCHMARK_OPENCLAW_CLI", str(cli_path))

    assert openclaw._runtime() == (node_bin, cli_path)


def test_summary_falls_back_to_current_checkout_after_host_move(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(json.dumps({"task_id": "remote-task", "domain": "test"}) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "effective_config.json").write_text(
        json.dumps({
            "config": {"task_manifest": "tasks.jsonl"},
            "repo": {"path": "/different/host/repo"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "REPO_ROOT", tmp_path)

    assert summarize._task_metadata(run_dir)["remote-task"]["domain"] == "test"
