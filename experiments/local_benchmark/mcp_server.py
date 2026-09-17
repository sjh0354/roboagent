"""Stdio MCP transport exposing only public simulator capabilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP, Image

from .io import load_manifest
from .simulator.world import StatefulWorld


def _load_world() -> StatefulWorld:
    manifest = os.environ["LOCAL_BENCHMARK_TASK_MANIFEST"]
    task_id = os.environ["LOCAL_BENCHMARK_TASK_ID"]
    tasks = {task["task_id"]: task for task in load_manifest(manifest)}
    if task_id not in tasks:
        raise RuntimeError(f"unknown task id: {task_id}")
    return StatefulWorld(
        tasks[task_id],
        seed=int(os.environ.get("LOCAL_BENCHMARK_SEED", "0")),
        output_dir=Path(os.environ["LOCAL_BENCHMARK_EPISODE_DIR"]),
        physical_success_prob=float(os.environ.get("LOCAL_BENCHMARK_PHYSICAL_SUCCESS_PROB", "0.60")),
        monitor_present=os.environ.get("LOCAL_BENCHMARK_MONITOR_PRESENT", "false").lower() == "true",
        base_single_attempt_prob=float(os.environ.get("LOCAL_BENCHMARK_BASE_SINGLE_ATTEMPT_PROB", "0.50")),
        integrated_monitor_retry_limit=int(os.environ.get("LOCAL_BENCHMARK_INTEGRATED_MONITOR_RETRY_LIMIT", "2")),
    )


WORLD = _load_world()
EPISODE_DIR = Path(os.environ["LOCAL_BENCHMARK_EPISODE_DIR"])
CALL_LOG = EPISODE_DIR / "mcp_tool_calls.jsonl"
PRIVATE_STATE = EPISODE_DIR / "final_private_state.json"


def _persist(tool: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
    with CALL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool, "arguments": arguments, "result": result}, ensure_ascii=False, default=str) + "\n")
    PRIVATE_STATE.write_text(
        json.dumps({"score": WORLD.score(), "public_state": WORLD.public_snapshot()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _content(result: Dict[str, Any]) -> List[Any]:
    blocks: List[Any] = [json.dumps(result, ensure_ascii=False, default=str)]
    observation = result.get("observation") or result
    paths = observation.get("rgb_paths") if isinstance(observation, dict) else None
    for path in (paths or [])[:2]:
        if path and Path(path).exists():
            blocks.append(Image(path=path))
    return blocks


mcp = FastMCP(
    "roboagent-local-simulator",
    instructions=(
        "Stateful synthetic physical environment. Hidden outcomes and private scoring are not exposed. "
        "Each physical goal resolves atomically in start_skill using the arm's aggregate probability. "
        "Monitoring/recovery is already integrated, so the same physical goal cannot be retried. "
        "Inspect returned RGB observations to determine semantic results."
    ),
    log_level="ERROR",
)


@mcp.tool()
def observe():
    """Observe public world state plus two current RGB views. No hidden success label is returned."""
    result = WORLD.observe()
    _persist("observe", {}, result)
    return _content(result)


@mcp.tool()
def start_skill(skill: str, arguments: Dict[str, Any]):
    """Resolve pick/place/navigate/pick_and_place once and return observable state/images."""
    result = WORLD.start_skill(skill, arguments)
    _persist("start_skill", {"skill": skill, "arguments": arguments}, result)
    return _content(result)


@mcp.tool()
def set_device(device: str, state: str):
    """Deterministically update a local simulated device. This never touches a real account or IoT device."""
    result = WORLD.set_device(device, state)
    _persist("set_device", {"device": device, "state": state}, result)
    return _content(result)


@mcp.tool()
def remember(text: str) -> str:
    """Store a user-stated preference in this isolated simulated session."""
    result = WORLD.store_preference(text)
    _persist("remember", {"text": text}, result)
    return json.dumps(result, ensure_ascii=False)


_persist("server_initialized", {}, {"diagnostic": "ready"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
