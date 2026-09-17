"""Loopback-only JSON transport for the native OpenClaw tool plugin."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .io import load_manifest
from .simulator.world import StatefulWorld


def _world_from_env() -> StatefulWorld:
    tasks = {task["task_id"]: task for task in load_manifest(os.environ["LOCAL_BENCHMARK_TASK_MANIFEST"])}
    task_id = os.environ["LOCAL_BENCHMARK_TASK_ID"]
    return StatefulWorld(
        tasks[task_id],
        seed=int(os.environ["LOCAL_BENCHMARK_SEED"]),
        output_dir=Path(os.environ["LOCAL_BENCHMARK_EPISODE_DIR"]),
        physical_success_prob=float(os.environ.get("LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB", "0.60")),
        monitor_present=os.environ.get("LOCAL_BENCHMARK_MONITOR_PRESENT", "false").lower() == "true",
        base_single_attempt_prob=float(os.environ.get("LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB", "0.50")),
        integrated_monitor_retry_limit=int(os.environ.get("LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT", "2")),
    )


WORLD = _world_from_env()
EPISODE_DIR = Path(os.environ["LOCAL_BENCHMARK_EPISODE_DIR"])
CALL_LOG = EPISODE_DIR / "openclaw_tool_calls.jsonl"
PRIVATE_STATE = EPISODE_DIR / "final_private_state.json"


def _persist(tool: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
    with CALL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool, "arguments": arguments, "result": result}, ensure_ascii=False) + "\n")
    PRIVATE_STATE.write_text(
        json.dumps({"score": WORLD.score(), "public_state": WORLD.public_snapshot()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "RoboAgentLocalBenchmark/1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _write(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._write(200, {"ready": True}) if self.path == "/health" else self._write(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            size = int(self.headers.get("Content-Length", "0"))
            arguments = json.loads(self.rfile.read(size) or b"{}")
            route = self.path.strip("/")
            if route == "observe":
                result = WORLD.observe()
            elif route == "start_skill":
                result = WORLD.start_skill(arguments.get("skill", ""), arguments.get("arguments", {}))
            elif route == "set_device":
                result = WORLD.set_device(arguments.get("device", ""), arguments.get("state"))
            elif route == "remember":
                result = WORLD.store_preference(arguments.get("text", ""))
            else:
                self._write(404, {"error": "unknown endpoint"})
                return
            _persist(route, arguments, result)
            self._write(200, result)
        except Exception as error:
            self._write(400, {"error": f"{type(error).__name__}: {error}"})


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ["LOCAL_BENCHMARK_HTTP_PORT"])
    _persist("server_initialized", {}, {"diagnostic": "ready"})
    (EPISODE_DIR / "http_service_ready.json").write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
