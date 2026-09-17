"""Launch the official Hermes AIAgent after synchronous MCP discovery.

Hermes v0.19.0's CLI oneshot constructs its tool snapshot before this local
MCP finishes dynamic registration.  This programmatic entry follows the
official AIAgent API but makes discovery an explicit barrier first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    episode_dir = Path(os.environ["LOCAL_BENCHMARK_EPISODE_DIR"])
    from tools.mcp_tool import discover_mcp_tools, shutdown_mcp_servers
    from run_agent import AIAgent

    discovered = discover_mcp_tools()
    (episode_dir / "hermes_discovered_tools.json").write_text(
        json.dumps(discovered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    expected = "mcp__local_benchmark__observe"
    if expected not in discovered:
        (episode_dir / "hermes_native_result.json").write_text(
            json.dumps({"error": f"required tool not discovered: {expected}", "discovered": discovered}, indent=2),
            encoding="utf-8",
        )
        shutdown_mcp_servers()
        return 3

    agent = AIAgent(
        api_key=os.environ["LOCAL_BENCHMARK_API_KEY"],
        base_url=os.environ["LOCAL_BENCHMARK_BASE_URL"],
        provider="custom",
        api_mode="chat_completions",
        model="kimi-k3",
        max_iterations=24,
        max_tokens=1600,
        enabled_toolsets=["mcp-local_benchmark"],
        save_trajectories=True,
        verbose_logging=True,
        quiet_mode=True,
        platform="cli",
        skip_context_files=True,
        load_soul_identity=False,
        ephemeral_system_prompt=(
            "This is an isolated simulated-physical benchmark. Use only mcp__local_benchmark__ tools. "
            "Do not use files, terminals, external services, or private evaluator state. Tool acceptance is not semantic success."
        ),
    )
    try:
        result = agent.run_conversation(os.environ["LOCAL_BENCHMARK_PROMPT"])
        (episode_dir / "hermes_native_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(result.get("final_response", ""))
        return 0 if not result.get("failed") else 4
    finally:
        agent.close()
        shutdown_mcp_servers()


if __name__ == "__main__":
    raise SystemExit(main())

