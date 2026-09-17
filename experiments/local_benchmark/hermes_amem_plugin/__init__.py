"""Hermes MemoryProvider bridge for the official A-MEM sidecar."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


class AMEMProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "amem"

    def is_available(self) -> bool:
        return bool(os.environ.get("LOCAL_BENCHMARK_AMEM_URL"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self.url = os.environ["LOCAL_BENCHMARK_AMEM_URL"].rstrip("/")

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.url + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=180) as response:
            return json.loads(response.read())

    def system_prompt_block(self) -> str:
        return "# A-MEM\nOfficial A-MEM owns long-term experience recall. Current task dialogue and static tool knowledge remain in Hermes."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        result = self._post("/prefetch", {"query": query, "session_id": session_id})
        rows = result.get("results", [])
        if not rows:
            return ""
        lines = [f"- {row.get('content', '')}" for row in rows]
        return "## Recalled A-MEM experience\n" + "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        self._post("/sync_turn", {"user": user_content, "assistant": assistant_content, "session_id": session_id})

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []


def register(ctx) -> None:
    ctx.register_memory_provider(AMEMProvider())
