"""Official Hermes Agent CLI integration through native stdio MCP."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

import yaml

from ..io import REPO_ROOT
from ..model import MODEL_NAME, aggregate_jsonl_usage, resolve_credentials


HERMES_COMMIT = "7651764ce63f44f4e02b1595798e73a67f678ebc"
HERMES_VERSION = "0.19.0"


def _hermes_runtime() -> tuple[Path, Path]:
    """Resolve the official Hermes CLI and interpreter without host-specific paths."""
    python_path = Path(
        os.environ.get(
            "LOCAL_BENCHMARK_HERMES_PYTHON",
            str(Path.home() / ".hermes/hermes-agent/venv/bin/python"),
        )
    ).expanduser()
    cli_candidates = [
        os.environ.get("LOCAL_BENCHMARK_HERMES_CLI"),
        shutil.which("hermes"),
        str(python_path.parent / "hermes"),
        str(Path.home() / ".local/bin/hermes"),
    ]
    cli = next((Path(value).expanduser() for value in cli_candidates if value and Path(value).expanduser().exists()), None)
    if cli is None:
        raise RuntimeError(
            "official Hermes CLI is not installed; set LOCAL_BENCHMARK_HERMES_CLI"
        )
    if not python_path.exists():
        raise RuntimeError(
            "official Hermes Python environment is not installed; set LOCAL_BENCHMARK_HERMES_PYTHON"
        )
    return cli, python_path


def _config(episode_dir: Path, base_url: str, use_amem: bool = False) -> Dict[str, Any]:
    return {
        "model": {
            "default": MODEL_NAME,
            "provider": "custom:local-benchmark",
            "context_length": 131072,
        },
        "providers": {
            "local-benchmark": {
                "name": "Local Benchmark kimi-k3",
                "base_url": base_url,
                "default_model": MODEL_NAME,
                "key_env": "LOCAL_BENCHMARK_API_KEY",
                "transport": "chat_completions",
                "max_output_tokens": 1600,
            }
        },
        "agent": {"max_turns": 24, "tool_use_enforcement": True},
        # Hermes registers each MCP server under the canonical dynamic
        # toolset name mcp-{server}.  Using the raw server alias as a CLI
        # --toolsets argument is validated before dynamic registration in
        # v0.19.0 and results in an empty schema list.
        "platform_toolsets": {"cli": ["mcp-local_benchmark"]},
        "display": {"show_reasoning": False, "show_cost": True},
        "memory": {
            # A-MEM owns long-term experience in its conditions; disabling
            # MEMORY.md here prevents duplicate retrieval of the same history.
            "memory_enabled": not use_amem,
            "user_profile_enabled": False,
            **({"provider": "amem"} if use_amem else {}),
        },
        "mcp_servers": {
            "local_benchmark": {
                # Keep the venv entry path intact. Resolving its symlink would
                # discard the venv's site-packages in the MCP child process.
                "command": str(REPO_ROOT / ".venv/bin/python"),
                "args": ["-m", "experiments.local_benchmark.mcp_server"],
                "env": {
                    "PYTHONPATH": str(REPO_ROOT),
                    "LOCAL_BENCHMARK_TASK_MANIFEST": os.environ["LOCAL_BENCHMARK_TASK_MANIFEST"],
                    "LOCAL_BENCHMARK_TASK_ID": os.environ["LOCAL_BENCHMARK_TASK_ID"],
                    "LOCAL_BENCHMARK_SEED": os.environ["LOCAL_BENCHMARK_SEED"],
                    "LOCAL_BENCHMARK_EPISODE_DIR": str(episode_dir.resolve()),
                    "LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB": os.environ["LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB"],
                    "LOCAL_BENCHMARK_MONITOR_PRESENT": os.environ["LOCAL_BENCHMARK_MONITOR_PRESENT"],
                    "LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB": os.environ["LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB"],
                    "LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT": os.environ["LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT"],
                },
                "timeout": 120,
                "connect_timeout": 60,
            }
        },
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, process: subprocess.Popen[str], timeout: float = 300.0) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"A-MEM service exited with {process.returncode}")
        try:
            with opener.open(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("A-MEM service did not become ready")


def run_hermes_episode(
    task: Dict[str, Any],
    seed: int,
    episode_dir: Path,
    physical_success_prob: float = 0.60,
    integrated_monitor_retry_limit: int = 2,
    timeout: int = 420,
    use_amem: bool = False,
    amem_store: Path | None = None,
    cam_mode: str = "none",
) -> Dict[str, Any]:
    _, hermes_python = _hermes_runtime()
    base_url, api_key = resolve_credentials()
    if not base_url or not api_key:
        raise RuntimeError("kimi-k3 credentials unavailable")
    episode_dir.mkdir(parents=True, exist_ok=True)
    if cam_mode not in {"none", "cam_adapted_symbolic"}:
        raise ValueError(f"unsupported CaM mode: {cam_mode}")
    hermes_home = episode_dir / "hermes_home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    if use_amem:
        plugin_target = hermes_home / "plugins/amem"
        shutil.copytree(REPO_ROOT / "experiments/local_benchmark/hermes_amem_plugin", plugin_target, dirs_exist_ok=True)
    manifest = os.environ.get("LOCAL_BENCHMARK_TASK_MANIFEST", str(REPO_ROOT / "experiments/local_benchmark/tasks/dev.jsonl"))
    env = os.environ.copy()
    # Hermes has a top-level ``utils`` package of its own. Do not let the
    # RoboAgent PYTHONPATH shadow it in the native runner process; the MCP
    # child receives its own explicit PYTHONPATH in _config().
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HERMES_HOME": str(hermes_home.resolve()),
            "LOCAL_BENCHMARK_API_KEY": api_key,
            "LOCAL_BENCHMARK_TASK_MANIFEST": str(Path(manifest).resolve()),
            "LOCAL_BENCHMARK_TASK_ID": task["task_id"],
            "LOCAL_BENCHMARK_SEED": str(seed),
            "LOCAL_BENCHMARK_EPISODE_DIR": str(episode_dir.resolve()),
            "LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB": str(physical_success_prob),
            "LOCAL_BENCHMARK_MONITOR_PRESENT": "true" if cam_mode != "none" else "false",
            "LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB": "0.50",
            "LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT": str(integrated_monitor_retry_limit),
        }
    )
    # Make env values available while building the isolated config.
    old = {key: os.environ.get(key) for key in env if key.startswith("LOCAL_BENCHMARK_")}
    os.environ.update({key: value for key, value in env.items() if key.startswith("LOCAL_BENCHMARK_")})
    try:
        config = _config(episode_dir, base_url, use_amem=use_amem)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prompt = (
        "You are being evaluated in a stateful synthetic physical environment. Complete this request: "
        + task["request"]
        + "\nYour FIRST response MUST be the mcp__local_benchmark__observe tool invocation with no prose. "
        "Use only tools whose names start mcp__local_benchmark__ for environment interaction. For a physical skill, call "
        "mcp__local_benchmark__start_skill with the exact argument schema. Each physical call resolves one normalized goal "
        f"atomically using the {'monitor-present 85%' if cam_mode != 'none' else 'no-monitor 60%'} aggregate policy. "
        "Inspect returned images/state. Monitoring/recovery and time effects are already folded into that probability, so "
        "do not retry the same physical goal. Do not read files or use shell commands. Never infer success from command acceptance."
    )
    workspace = episode_dir / "agent_workspace"
    workspace.mkdir(exist_ok=True)
    env["LOCAL_BENCHMARK_BASE_URL"] = base_url
    env["LOCAL_BENCHMARK_PROMPT"] = prompt
    amem_process = None
    amem_stdout = None
    amem_stderr = None
    if use_amem:
        port = _free_port()
        amem_url = f"http://127.0.0.1:{port}"
        env.update({
            "LOCAL_BENCHMARK_AMEM_URL": amem_url,
            "LOCAL_BENCHMARK_AMEM_PORT": str(port),
            "LOCAL_BENCHMARK_AMEM_STORE": str((amem_store or (episode_dir / "amem_store.json")).resolve()),
            "LOCAL_BENCHMARK_AMEM_EVENT_LOG": str((episode_dir / "amem_events.jsonl").resolve()),
            "LOCAL_BENCHMARK_AMEM_MODEL_LOG": str((episode_dir / "amem_model_calls.jsonl").resolve()),
            "HF_HOME": str((REPO_ROOT / ".runtime/huggingface").resolve()),
            "TOKENIZERS_PARALLELISM": "false",
        })
        amem_stdout = (episode_dir / "amem_service_stdout.txt").open("w", encoding="utf-8")
        amem_stderr = (episode_dir / "amem_service_stderr.txt").open("w", encoding="utf-8")
        amem_process = subprocess.Popen(
            [str(REPO_ROOT / ".venv/bin/python"), "-m", "experiments.local_benchmark.amem_service"],
            cwd=REPO_ROOT, env=env, text=True, stdout=amem_stdout, stderr=amem_stderr,
        )
        _wait_http(amem_url, amem_process)
    command = [
        str(hermes_python),
        str(REPO_ROOT / "experiments/local_benchmark/hermes_native_runner.py"),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=workspace, env=env, text=True, capture_output=True, timeout=timeout)
        (episode_dir / "hermes_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (episode_dir / "hermes_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if amem_process:
            # Hermes drains memory-provider writes for five seconds. Give an
            # already accepted A-MEM request a bounded grace period to finish.
            deadline = time.monotonic() + 30
            event_log = episode_dir / "amem_events.jsonl"
            while time.monotonic() < deadline:
                if event_log.exists() and '"event": "write"' in event_log.read_text(encoding="utf-8"):
                    break
                if amem_process.poll() is not None:
                    break
                time.sleep(0.25)
    finally:
        if amem_process:
            amem_process.terminate()
            try:
                amem_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                amem_process.kill()
            amem_stdout.close()
            amem_stderr.close()
    (episode_dir / "hermes_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (episode_dir / "hermes_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    private_path = episode_dir / "final_private_state.json"
    private = json.loads(private_path.read_text(encoding="utf-8")) if private_path.exists() else {"score": {"task_success": False, "checks": [], "counts": {}}}
    native_result_path = episode_dir / "hermes_native_result.json"
    native_result = json.loads(native_result_path.read_text(encoding="utf-8")) if native_result_path.exists() else {}
    usage = {
        key: native_result.get(key)
        for key in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "total_tokens", "api_calls", "estimated_cost_usd", "cost_status", "cost_source", "session_id",
        )
        if key in native_result
    }
    amem_usage = aggregate_jsonl_usage(episode_dir / "amem_model_calls.jsonl")
    cam_generation_usage = aggregate_jsonl_usage(episode_dir / "cam_adapted/generation_model_call.jsonl")
    tool_log = episode_dir / "mcp_tool_calls.jsonl"
    tool_call_count = len(tool_log.read_text(encoding="utf-8").splitlines()) - 1 if tool_log.exists() else 0
    framework_ok = completed.returncode == 0 and tool_call_count > 0
    return {
        "method_id": "hermes_amem_cam" if use_amem and cam_mode != "none" else "hermes_amem" if use_amem else "hermes_cam" if cam_mode != "none" else "hermes_native",
        "framework": "NousResearch/hermes-agent native AIAgent loop",
        "framework_version": HERMES_VERSION,
        "framework_commit": HERMES_COMMIT,
        "memory_backend": "official agiresearch/A-mem" if use_amem else "Hermes native",
        "amem_commit": "ceffb860f0712bbae97b184d440df62bc910ca8d" if use_amem else None,
        "monitor_backend": "aggregate_monitor_assumption_85" if cam_mode != "none" else "none",
        "execution_control_policy": "direct_aggregate_bernoulli",
        "monitor_present": cam_mode != "none",
        "monitor_effect_integrated": cam_mode != "none",
        "effective_physical_success_probability": physical_success_prob,
        "explicit_physical_retries_simulated": False,
        "seed_controls_physics": False,
        "model": MODEL_NAME,
        "task_id": task["task_id"],
        "seed": seed,
        "simulated_physical_execution": True,
        "status": "completed" if framework_ok else "framework_error",
        "returncode": completed.returncode,
        "final_response": native_result.get("final_response") or completed.stdout.strip(),
        "native_turn_exit_reason": native_result.get("turn_exit_reason"),
        "score": private["score"],
        "tool_calls": tool_call_count,
        "usage": usage,
        "amem_usage": amem_usage,
        "cam_generation_usage": cam_generation_usage,
        "wall_seconds": time.monotonic() - started,
    }
