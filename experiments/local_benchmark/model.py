"""Strict kimi-k3 OpenAI-compatible model gateway with auditable usage logs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MODEL_NAME = "kimi-k3"


def aggregate_jsonl_usage(path: Path) -> Dict[str, int]:
    """Aggregate one-call-per-record usage logs into a common schema."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }
    path = Path(path)
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        usage = record.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        if usage or record.get("api_call_attempt"):
            totals["api_calls"] += 1
        if not usage:
            continue
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("input") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("output") or 0)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cache_read_tokens"] += int(
            usage.get("cache_read_tokens") or usage.get("cacheRead") or prompt_details.get("cached_tokens") or 0
        )
        totals["cache_write_tokens"] += int(usage.get("cache_write_tokens") or usage.get("cacheWrite") or 0)
        totals["reasoning_tokens"] += int(
            usage.get("reasoning_tokens") or completion_details.get("reasoning_tokens") or 0
        )
        totals["total_tokens"] += int(usage.get("total_tokens") or usage.get("totalTokens") or input_tokens + output_tokens)
    return totals


def parse_reference_script(path: str) -> Dict[str, str]:
    """Read only literal url/key assignments from the provided reference script.

    The shell file is never executed.  This avoids side effects and keeps secrets
    out of tracked configuration and command output.
    """
    values: Dict[str, str] = {}
    reference = Path(path)
    if not reference.exists():
        return values
    for raw_line in reference.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"url", "AIGW_API_KEY"}:
            continue
        value = value.strip().strip("\"").strip("'")
        if "$" not in value and "`" not in value:
            values[key] = value
    return values


def resolve_credentials() -> tuple[str, str]:
    reference = os.getenv(
        "LOCAL_BENCHMARK_CREDENTIALS_SCRIPT",
        str(Path.home() / "Localwork/test.sh"),
    )
    parsed = parse_reference_script(reference)
    base_url = (
        os.getenv("LOCAL_BENCHMARK_BASE_URL")
        or parsed.get("url")
        or os.getenv("PLANNER_VLM_BASE_URL")
        or ""
    ).rstrip("/")
    api_key = (
        os.getenv("LOCAL_BENCHMARK_API_KEY")
        or parsed.get("AIGW_API_KEY")
        or os.getenv("AIGW_API_KEY")
        or os.getenv("PLANNER_VLM_API_KEY")
        or ""
    )
    return base_url, api_key


def configure_roboagent_model_environment() -> None:
    base_url, api_key = resolve_credentials()
    if not base_url or not api_key:
        raise RuntimeError("kimi-k3 credentials unavailable; set LOCAL_BENCHMARK_BASE_URL and LOCAL_BENCHMARK_API_KEY")
    os.environ["PLANNER_VLM_PROVIDER"] = "openai_compatible"
    os.environ["PLANNER_VLM_BASE_URL"] = base_url
    os.environ["PLANNER_VLM_API_KEY"] = api_key
    os.environ["PI0_ACTION_MONITOR_PROVIDER"] = "openai_compatible"
    os.environ["PI0_ACTION_MONITOR_BASE_URL"] = base_url
    os.environ["PI0_ACTION_MONITOR_API_KEY"] = api_key
    os.environ["PI0_ACTION_MONITOR_MODEL"] = MODEL_NAME
    # A fallback would silently violate the experiment protocol.
    os.environ.pop("PLANNER_VLM_FALLBACK_MODEL", None)


def _data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class ModelGateway:
    def __init__(self, log_path: Path, timeout_seconds: float = 180.0):
        self.base_url, self.api_key = resolve_credentials()
        if not self.base_url or not self.api_key:
            raise RuntimeError("kimi-k3 credentials unavailable")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.call_count = 0
        self.role_counts: Dict[str, int] = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _write_log(self, record: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _post_json(self, payload: Dict[str, Any], role: str, phase: str = "initial") -> tuple[Dict[str, Any], float]:
        """POST with a bounded, fully logged retry policy for transport failures."""
        for attempt in range(1, 4):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.call_count += 1
                self.role_counts[role] = self.role_counts.get(role, 0) + 1
                return body, time.monotonic() - started
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:2000]
                message = f"HTTP {error.code}: {detail}"
                retryable = error.code in {429, 500, 502, 503, 504}
            except (urllib.error.URLError, TimeoutError) as error:
                message = f"{type(error).__name__}: {error}"
                retryable = True
            self.call_count += 1
            self.role_counts[role] = self.role_counts.get(role, 0) + 1
            self._write_log({
                "role": role, "model": MODEL_NAME, "phase": phase, "attempt": attempt,
                "api_call_attempt": True, "error": message, "retryable": retryable,
                "wall_seconds": time.monotonic() - started,
            })
            if not retryable or attempt == 3:
                raise RuntimeError(f"kimi-k3 request failed: {message}")
            # The gateway occasionally closes TLS connections in short bursts;
            # keep the retry count fixed but give it time to recover.
            time.sleep(float(5 * attempt))
        raise AssertionError("unreachable")

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        image_paths: Optional[Iterable[str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        images = [str(Path(path).resolve()) for path in (image_paths or [])]
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in images:
            content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_completion_tokens": int(max_tokens),
            "temperature": float(temperature),
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        body, elapsed = self._post_json(payload, role)
        usage = body.get("usage") or {}
        for key in self.usage:
            self.usage[key] += int(usage.get(key) or 0)
        text = body["choices"][0]["message"].get("content") or ""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            fallback = None
            if role == "online_monitor":
                decisions = re.findall(r"decision\s*[:=]\s*[\"']?(continue|complete|stuck)", text, flags=re.IGNORECASE)
                if decisions:
                    decision = decisions[-1].lower()
                    fallback = {
                        "decision": decision,
                        "recommended_stop": decision in {"complete", "stuck"},
                        "confidence": "unparsed",
                        "reason": "Recovered the explicit decision from a truncated natural-language monitor response.",
                        "parse_fallback": "explicit_decision_text",
                    }
                else:
                    fallback = {
                        "decision": "continue",
                        "recommended_stop": False,
                        "confidence": "low",
                        "reason": "Monitor response was truncated before a machine-readable decision; fail-safe policy continues to the common endpoint.",
                        "parse_fallback": "truncation_continue_failsafe",
                    }
            self._write_log(
                {
                    "role": role,
                    "model": body.get("model") or MODEL_NAME,
                    "raw_response": text,
                    "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
                    "usage": usage,
                    "api_call_attempt": True,
                    "wall_seconds": elapsed,
                    "parse_error": "response was not complete JSON",
                    "recovered_response": fallback,
                }
            )
            if fallback is not None:
                return fallback
            raise
        self._write_log(
            {
                "role": role,
                "model": body.get("model") or MODEL_NAME,
                "request": {
                    "system": system,
                    "prompt": prompt,
                    "images": [
                        {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                        for path in images
                    ],
                },
                "response": result,
                "usage": usage,
                "api_call_attempt": True,
                "wall_seconds": elapsed,
            }
        )
        return result

    def complete_tool_json(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        tool_name: str,
        schema: Dict[str, Any],
        image_paths: Optional[Iterable[str]] = None,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Force a structured function call for models that over-explain JSON."""
        images = [str(Path(path).resolve()) for path in (image_paths or [])]
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in images:
            content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "tools": [{"type": "function", "function": {"name": tool_name, "description": "Submit the generated monitor", "parameters": schema}}],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "max_completion_tokens": int(max_tokens),
            "temperature": 0.0,
            "stream": False,
        }
        started = time.monotonic()
        body, _ = self._post_json(payload, role)
        usage = body.get("usage") or {}
        for key in self.usage:
            self.usage[key] += int(usage.get(key) or 0)
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            self._write_log({"role": role, "model": body.get("model") or MODEL_NAME, "raw_response": message, "usage": usage, "api_call_attempt": True, "wall_seconds": time.monotonic() - started, "parse_error": "required tool call missing; issuing one bounded continuation"})
            continuation = dict(payload)
            continuation["messages"] = payload["messages"] + [
                {"role": "assistant", "content": message.get("content") or ""},
                {"role": "user", "content": "Conclude now. Call the required function immediately; provide no more analysis."},
            ]
            # kimi-k3 may emit a long private-style analysis in ``content``
            # before honoring a forced tool call.  Give the one permitted
            # continuation the same bounded budget as the initial request;
            # both calls remain separately logged and charged.
            continuation["max_completion_tokens"] = int(max_tokens)
            follow_body, _ = self._post_json(continuation, role, phase="continuation")
            follow_usage = follow_body.get("usage") or {}
            for key in self.usage:
                self.usage[key] += int(follow_usage.get(key) or 0)
            message = follow_body["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                self._write_log({"role": role, "model": follow_body.get("model") or MODEL_NAME, "raw_response": message, "usage": follow_usage, "api_call_attempt": True, "parse_error": "required tool call missing after bounded continuation"})
                raise ValueError("kimi-k3 did not return the required structured tool call")
            # The initial request was already logged above as one API call.
            # Store only this continuation's usage on the successful record so
            # downstream aggregation cannot double count it.
            usage = follow_usage
        result = json.loads(calls[0]["function"]["arguments"])
        self._write_log({
            "role": role, "model": body.get("model") or MODEL_NAME,
            "request": {"system": system, "prompt": prompt, "tool_name": tool_name, "schema": schema, "images": [
                {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in images
            ]},
            "response": result, "usage": usage, "api_call_attempt": True, "wall_seconds": time.monotonic() - started,
        })
        return result

    def monitor(self, instruction: str, frames: List[str], elapsed_seconds: float) -> Dict[str, Any]:
        return self.complete_json(
            role="online_monitor",
            system="You are a visual execution monitor. Use only the ordered images. Never assume the command succeeded.",
            prompt=(
                f"Instruction: {instruction}\nElapsed simulated seconds: {elapsed_seconds:.1f}. "
                "Images are chronological. Return JSON with decision (continue|complete|stuck), recommended_stop (boolean), "
                "confidence, and reason. Stop only on visible completion, visible stall, or unsafe divergence."
            ),
            image_paths=frames,
            max_tokens=900,
        )
