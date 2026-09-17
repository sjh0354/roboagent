"""Isolated HTTP sidecar around the pinned official A-MEM implementation."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from agentic_memory.llm_controller import OpenAIController
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


STORE_PATH = Path(os.environ["LOCAL_BENCHMARK_AMEM_STORE"])
EVENT_LOG = Path(os.environ["LOCAL_BENCHMARK_AMEM_EVENT_LOG"])
MODEL_LOG = Path(os.environ["LOCAL_BENCHMARK_AMEM_MODEL_LOG"])


def _append(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class TracedOpenAIController(OpenAIController):
    """A transport-only override that preserves A-MEM's prompts and parsing."""

    def __init__(self) -> None:
        self.model = "kimi-k3"
        self.client = OpenAI(
            api_key=os.environ["LOCAL_BENCHMARK_API_KEY"],
            base_url=os.environ["LOCAL_BENCHMARK_BASE_URL"],
        )

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000,
        )
        usage = response.usage.model_dump() if response.usage else None
        content = response.choices[0].message.content or ""
        _append(MODEL_LOG, {"model": self.model, "prompt": prompt, "response": content, "usage": usage})
        return content


def _serialize_note(note: MemoryNote) -> Dict[str, Any]:
    return {key: getattr(note, key) for key in (
        "content", "id", "keywords", "links", "retrieval_count", "timestamp", "last_accessed",
        "context", "evolution_history", "category", "tags",
    )}


class Store:
    def __init__(self) -> None:
        self.system = AgenticMemorySystem(
            model_name=os.environ.get("LOCAL_BENCHMARK_AMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_backend="openai", llm_model="kimi-k3",
            api_key=os.environ["LOCAL_BENCHMARK_API_KEY"],
        )
        self.system.llm_controller.llm = TracedOpenAIController()
        self._load()

    def _load(self) -> None:
        if not STORE_PATH.exists():
            return
        for payload in json.loads(STORE_PATH.read_text(encoding="utf-8")):
            note = MemoryNote(**payload)
            self.system.memories[note.id] = note
            self.system.retriever.add_document(note.content, _serialize_note(note), note.id)
        _append(EVENT_LOG, {"event": "load", "notes": len(self.system.memories)})

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(
            json.dumps([_serialize_note(note) for note in self.system.memories.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, content: str) -> Dict[str, Any]:
        analysis = self.system.analyze_content(content)
        note_id = self.system.add_note(content, **analysis)
        self._save()
        note = self.system.read(note_id)
        result = {"id": note_id, "note": _serialize_note(note), "total_notes": len(self.system.memories)}
        _append(EVENT_LOG, {"event": "write", "content": content, "result": result})
        return result

    def search(self, query: str) -> Dict[str, Any]:
        results = self.system.search_agentic(query, k=5)
        result = {"query": query, "results": results, "retrieval_chars": len(json.dumps(results, ensure_ascii=False))}
        _append(EVENT_LOG, {"event": "retrieve", **result})
        return result


STORE = Store()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _reply(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._reply(200, {"ready": True, "notes": len(STORE.system.memories)}) if self.path == "/health" else self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/prefetch":
                result = STORE.search(str(payload.get("query", "")))
            elif self.path == "/sync_turn":
                content = "USER: " + str(payload.get("user", "")) + "\nASSISTANT: " + str(payload.get("assistant", ""))
                result = STORE.add(content)
            else:
                self._reply(404, {"error": "not found"})
                return
            self._reply(200, result)
        except Exception as error:
            _append(EVENT_LOG, {"event": "error", "error": f"{type(error).__name__}: {error}"})
            self._reply(500, {"error": f"{type(error).__name__}: {error}"})


def main() -> None:
    port = int(os.environ["LOCAL_BENCHMARK_AMEM_PORT"])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
