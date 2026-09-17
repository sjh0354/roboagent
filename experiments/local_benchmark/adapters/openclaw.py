"""Official OpenClaw native plugin and embedded-agent integration."""

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

from ..io import REPO_ROOT
from ..model import MODEL_NAME, resolve_credentials


OPENCLAW_VERSION = "2026.7.1"
OPENCLAW_COMMIT = "2d2ddc4"


def _runtime() -> tuple[Path, Path]:
    bundled_node_bin = REPO_ROOT / ".runtime/node24/node_modules/.bin"
    system_node = shutil.which("node")
    node_bin = Path(
        os.environ.get(
            "LOCAL_BENCHMARK_NODE_BIN_DIR",
            str(bundled_node_bin if bundled_node_bin.exists() else Path(system_node).parent if system_node else bundled_node_bin),
        )
    ).expanduser()
    bundled_cli = REPO_ROOT / ".runtime/openclaw-2026.7.1/node_modules/.bin/openclaw"
    cli = Path(
        os.environ.get(
            "LOCAL_BENCHMARK_OPENCLAW_CLI",
            str(bundled_cli if bundled_cli.exists() else shutil.which("openclaw") or bundled_cli),
        )
    ).expanduser()
    if not node_bin.exists():
        raise RuntimeError(
            "Node.js runtime is not installed; set LOCAL_BENCHMARK_NODE_BIN_DIR"
        )
    if not cli.exists():
        raise RuntimeError(
            "official OpenClaw 2026.7.1 runtime is not installed; set LOCAL_BENCHMARK_OPENCLAW_CLI"
        )
    return node_bin, cli


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _config(episode_dir: Path, base_url: str) -> Dict[str, Any]:
    plugin = REPO_ROOT / "experiments/local_benchmark/openclaw_plugin"
    workspace = episode_dir / "agent_workspace"
    workspace.mkdir(exist_ok=True)
    return {
        "agents": {"defaults": {"workspace": str(workspace.resolve()), "model": {"primary": "local-benchmark/kimi-k3"}, "timeoutSeconds": 360}},
        "models": {"mode": "replace", "providers": {"local-benchmark": {
            "baseUrl": base_url,
            "apiKey": "${LOCAL_BENCHMARK_API_KEY}",
            "api": "openai-completions",
            "timeoutSeconds": 180,
            "models": [{
                "id": MODEL_NAME, "name": "Kimi K3", "input": ["text", "image"],
                "contextWindow": 131072, "maxTokens": 1600,
                "compat": {"supportsDeveloperRole": False},
            }],
        }}},
        "plugins": {
            "enabled": True,
            "allow": ["local-benchmark"],
            "load": {"paths": [str(plugin.resolve())]},
            "entries": {"local-benchmark": {"enabled": True}},
        },
        "tools": {"allow": [
            "local_benchmark_observe", "local_benchmark_start_skill", "local_benchmark_set_device",
            "local_benchmark_remember",
        ]},
    }


def _wait_ready(url: str, process: subprocess.Popen[str], timeout: float = 20.0) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"simulator service exited with {process.returncode}")
        try:
            with opener.open(url + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("simulator service did not become ready")


def _session_usage(session_path: Path) -> Dict[str, int]:
    totals = {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "api_calls": 0,
    }
    if not session_path.exists():
        return totals
    for line in session_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("message") or {}
        if record.get("type") != "message" or message.get("role") != "assistant" or message.get("model") != MODEL_NAME:
            continue
        usage = message.get("usage") or {}
        if not usage:
            continue
        totals["api_calls"] += 1
        totals["input_tokens"] += int(usage.get("input") or 0)
        totals["output_tokens"] += int(usage.get("output") or 0)
        totals["cache_read_tokens"] += int(usage.get("cacheRead") or 0)
        totals["cache_write_tokens"] += int(usage.get("cacheWrite") or 0)
        totals["reasoning_tokens"] += int(usage.get("reasoningTokens") or 0)
        totals["total_tokens"] += int(usage.get("totalTokens") or 0)
    return totals


def run_openclaw_episode(
    task: Dict[str, Any], seed: int, episode_dir: Path,
    physical_success_prob: float = 0.60, integrated_monitor_retry_limit: int = 2, timeout: int = 480,
) -> Dict[str, Any]:
    node_bin, cli = _runtime()
    base_url, api_key = resolve_credentials()
    if not base_url or not api_key:
        raise RuntimeError("kimi-k3 credentials unavailable")
    episode_dir.mkdir(parents=True, exist_ok=True)
    state_dir = episode_dir / "openclaw_state"
    state_dir.mkdir(exist_ok=True)
    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    manifest = os.environ.get("LOCAL_BENCHMARK_TASK_MANIFEST", str(REPO_ROOT / "experiments/local_benchmark/tasks/dev.jsonl"))
    env = os.environ.copy()
    env.update({
        "PATH": str(node_bin.resolve()) + os.pathsep + env.get("PATH", ""),
        "OPENCLAW_STATE_DIR": str(state_dir.resolve()),
        "OPENCLAW_CONFIG_PATH": str((state_dir / "openclaw.json").resolve()),
        "LOCAL_BENCHMARK_API_KEY": api_key,
        "LOCAL_BENCHMARK_HTTP_URL": endpoint,
        "LOCAL_BENCHMARK_HTTP_PORT": str(port),
        "LOCAL_BENCHMARK_TASK_MANIFEST": str(Path(manifest).resolve()),
        "LOCAL_BENCHMARK_TASK_ID": task["task_id"],
        "LOCAL_BENCHMARK_SEED": str(seed),
        "LOCAL_BENCHMARK_EPISODE_DIR": str(episode_dir.resolve()),
        "LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB": str(physical_success_prob),
        "LOCAL_BENCHMARK_MONITOR_PRESENT": "false",
        "LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB": "0.50",
        "LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT": str(integrated_monitor_retry_limit),
        "PYTHONPATH": str(REPO_ROOT),
    })
    (state_dir / "openclaw.json").write_text(json.dumps(_config(episode_dir, base_url), indent=2), encoding="utf-8")
    prompt = (
        "You are evaluated in an isolated stateful synthetic physical environment. Complete: " + task["request"]
        + "\nYour first action must be local_benchmark_observe. Use only local_benchmark_* tools. "
        "Each physical start_skill call resolves one normalized goal atomically using the no-monitor aggregate 60% policy. "
        "Inspect returned state/images. Monitoring, self-interruption, and time effects are already folded into that probability, "
        "so do not retry the same physical goal. Tool acceptance never proves semantic success. "
        "Do not read files, run shell commands, or access external services."
    )
    service_out = (episode_dir / "http_service_stdout.txt").open("w", encoding="utf-8")
    service_err = (episode_dir / "http_service_stderr.txt").open("w", encoding="utf-8")
    service = subprocess.Popen(
        [str(REPO_ROOT / ".venv/bin/python"), "-m", "experiments.local_benchmark.http_service"],
        cwd=REPO_ROOT, env=env, text=True, stdout=service_out, stderr=service_err,
    )
    completed: subprocess.CompletedProcess[str]
    started = time.monotonic()
    try:
        _wait_ready(endpoint, service)
        command = [
            str(cli), "agent", "--local", "--json", "--model", "local-benchmark/kimi-k3",
            "--session-id", f"benchmark-{task['task_id']}-{seed}", "--timeout", "420", "--message", prompt,
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()
        service_out.close()
        service_err.close()
    (episode_dir / "openclaw_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (episode_dir / "openclaw_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    try:
        native = json.loads(completed.stdout)
    except json.JSONDecodeError:
        native = {"raw_stdout": completed.stdout}
    (episode_dir / "openclaw_native_result.json").write_text(json.dumps(native, ensure_ascii=False, indent=2), encoding="utf-8")
    session_value = (((native.get("meta") or {}).get("agentMeta") or {}).get("sessionFile"))
    usage = _session_usage(Path(session_value)) if session_value else {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "api_calls": 0,
    }
    private_path = episode_dir / "final_private_state.json"
    private = json.loads(private_path.read_text(encoding="utf-8")) if private_path.exists() else {"score": {"task_success": False, "checks": [], "counts": {}}}
    tool_log = episode_dir / "openclaw_tool_calls.jsonl"
    tool_calls = max(0, len(tool_log.read_text(encoding="utf-8").splitlines()) - 1) if tool_log.exists() else 0
    return {
        "method_id": "openclaw_native",
        "framework": "openclaw/openclaw native embedded agent + native tool plugin",
        "framework_version": OPENCLAW_VERSION,
        "framework_commit": OPENCLAW_COMMIT,
        "execution_control_policy": "direct_aggregate_bernoulli",
        "monitor_present": False,
        "monitor_effect_integrated": False,
        "effective_physical_success_probability": physical_success_prob,
        "explicit_physical_retries_simulated": False,
        "seed_controls_physics": False,
        "model": MODEL_NAME,
        "task_id": task["task_id"], "seed": seed,
        "simulated_physical_execution": True,
        "status": "completed" if completed.returncode == 0 and tool_calls > 0 else "framework_error",
        "returncode": completed.returncode,
        "score": private["score"], "tool_calls": tool_calls, "usage": usage,
        "native_result": native,
        "wall_seconds": time.monotonic() - started,
    }
