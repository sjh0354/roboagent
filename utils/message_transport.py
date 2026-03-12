"""
Transport abstractions for robot input/output channels.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import requests

try:
    from utils.funasr_manager import FunASRManager
except ImportError:
    FunASRManager = None  # type: ignore[assignment]


@dataclass
class TransportMessage:
    text: str
    message_id: Optional[str] = None
    sender_id: Optional[str] = None
    sender_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportSendResult:
    success: bool
    feedback: str
    message_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MessageTransport:
    """Base transport used by planners and executors."""

    def start(self):
        """Start background listeners if needed."""

    def stop(self):
        """Stop background listeners if needed."""

    def get_command(self) -> Optional[str]:
        """Return a new task command when available."""
        return None

    def get_response(self) -> Optional[str]:
        """Return a free-form response while the planner is waiting."""
        return None

    def send_message(
        self,
        message: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransportSendResult:
        return TransportSendResult(
            success=False,
            feedback="Transport does not support outbound messaging",
            error="unsupported",
        )

    def can_send_messages(self) -> bool:
        return False

    def should_mirror_local_speech(self) -> bool:
        return False


class VoiceTransport(MessageTransport):
    """Compatibility transport that keeps the existing ASR-based flow."""

    def __init__(self, asr: Optional["FunASRManager"] = None, verbose: bool = True):
        self.verbose = verbose
        self.asr = asr or (FunASRManager(verbose=verbose) if FunASRManager else None)

    def start(self):
        if self.asr:
            self.asr.start()

    def stop(self):
        if self.asr:
            self.asr.stop()

    def get_command(self) -> Optional[str]:
        if not self.asr:
            return None
        return self.asr.get_command()

    def get_response(self) -> Optional[str]:
        if not self.asr:
            return None
        return self.asr.get_speech()


class OpenClawLarkTransport(MessageTransport):
    """
    Polls a local OpenClaw gateway through the CLI for Feishu/Lark messages.

    This transport intentionally reuses the user's existing OpenClaw channel
    configuration instead of reimplementing the Feishu SDK in this repository.
    """

    def __init__(
        self,
        target: str,
        account_id: Optional[str] = None,
        channel: str = "feishu",
        poll_interval: float = 1.0,
        history_limit: int = 20,
        binary: str = "openclaw",
        verbose: bool = True,
    ):
        if not target:
            raise ValueError("OpenClaw Lark transport requires a target chat/group id")

        self.target = target
        self.account_id = account_id
        self.channel = channel
        self.poll_interval = max(0.2, poll_interval)
        self.history_limit = max(1, history_limit)
        self.binary = binary
        self.verbose = verbose

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._incoming: "queue.Queue[TransportMessage]" = queue.Queue()
        self._last_seen_message_id: Optional[str] = None
        self._ignored_message_ids: set[str] = set()
        self._recent_sent_texts: Deque[str] = deque(maxlen=20)

    def start(self):
        if self._running:
            return
        self._prime_last_seen_message_id()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(
                f"🌐 OpenClaw Lark transport started "
                f"(channel={self.channel}, target={self.target}, account={self.account_id or 'default'})"
            )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.verbose:
            print("🛑 OpenClaw Lark transport stopped")

    def get_command(self) -> Optional[str]:
        message = self._get_next_message()
        return message.text if message else None

    def get_response(self) -> Optional[str]:
        message = self._get_next_message()
        return message.text if message else None

    def send_message(
        self,
        message: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransportSendResult:
        if not message:
            return TransportSendResult(success=False, feedback="Empty message", error="empty_message")

        target = recipient or self.target
        command = [
            self.binary,
            "message",
            "send",
            "--channel",
            self.channel,
            "--target",
            target,
            "--message",
            message,
            "--json",
        ]
        if self.account_id:
            command.extend(["--account", self.account_id])

        completed = self._run_openclaw(command)
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip() or "openclaw send failed"
            return TransportSendResult(success=False, feedback="OpenClaw send failed", error=error)

        payload = self._safe_json_loads(completed.stdout)
        message_id = self._extract_message_id(payload)
        if message_id:
            self._ignored_message_ids.add(message_id)
        self._recent_sent_texts.append(message.strip())
        return TransportSendResult(
            success=True,
            feedback=f"Message sent via OpenClaw {self.channel}",
            message_id=message_id,
            raw=payload,
        )

    def can_send_messages(self) -> bool:
        return True

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_once()
            except Exception as error:  # pragma: no cover - defensive guard
                if self.verbose:
                    print(f"⚠️  OpenClaw Lark poll error: {error}")
            time.sleep(self.poll_interval)

    def _poll_once(self):
        command = [
            self.binary,
            "message",
            "read",
            "--channel",
            self.channel,
            "--target",
            self.target,
            "--limit",
            str(self.history_limit),
            "--json",
        ]
        if self.account_id:
            command.extend(["--account", self.account_id])
        if self._last_seen_message_id:
            command.extend(["--after", self._last_seen_message_id])

        completed = self._run_openclaw(command)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "openclaw read failed")

        payload = self._safe_json_loads(completed.stdout)
        messages = self._extract_messages(payload)
        for message in messages:
            if not message.text.strip():
                continue
            if message.message_id:
                self._last_seen_message_id = message.message_id
                if message.message_id in self._ignored_message_ids:
                    self._ignored_message_ids.discard(message.message_id)
                    continue
            if self._looks_like_recent_self_echo(message.text):
                continue
            self._incoming.put(message)

    def _prime_last_seen_message_id(self):
        command = [
            self.binary,
            "message",
            "read",
            "--channel",
            self.channel,
            "--target",
            self.target,
            "--limit",
            "1",
            "--json",
        ]
        if self.account_id:
            command.extend(["--account", self.account_id])
        completed = self._run_openclaw(command)
        if completed.returncode != 0:
            return
        payload = self._safe_json_loads(completed.stdout)
        messages = self._extract_messages(payload)
        if messages and messages[-1].message_id:
            self._last_seen_message_id = messages[-1].message_id

    def _get_next_message(self) -> Optional[TransportMessage]:
        try:
            return self._incoming.get_nowait()
        except queue.Empty:
            return None

    def _run_openclaw(self, command: List[str]) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )

    def _looks_like_recent_self_echo(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return True
        return normalized in self._recent_sent_texts

    def _safe_json_loads(self, raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()
        if not text:
            return {}
        start = text.find("{")
        if start == -1:
            start = text.find("[")
        if start > 0:
            text = text[start:]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw_output": raw_output}
        return parsed if isinstance(parsed, dict) else {"items": parsed}

    def _extract_messages(self, payload: Dict[str, Any]) -> List[TransportMessage]:
        candidates = self._find_message_candidates(payload)
        messages: List[TransportMessage] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            message_id = self._first_non_empty(
                item.get("message_id"),
                item.get("messageId"),
                item.get("id"),
            )
            sender_id = self._extract_sender_id(item)
            text = self._extract_text(item)
            if not text:
                continue
            messages.append(
                TransportMessage(
                    text=text,
                    message_id=message_id,
                    sender_id=sender_id,
                    raw=item,
                )
            )
        return messages

    def _find_message_candidates(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("messages", "items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = self._find_message_candidates(value)
                if nested:
                    return nested
        if any(key in payload for key in ("message_id", "messageId", "content", "text", "body")):
            return [payload]
        return []

    def _extract_text(self, item: Dict[str, Any]) -> str:
        direct_text = item.get("text")
        if isinstance(direct_text, str):
            return direct_text.strip()

        body = item.get("body")
        if isinstance(body, dict):
            body_text = body.get("text")
            if isinstance(body_text, str):
                return body_text.strip()
            body_content = body.get("content")
            if isinstance(body_content, str):
                return self._extract_text_from_content(body_content)

        content = item.get("content")
        if isinstance(content, str):
            return self._extract_text_from_content(content)
        if isinstance(content, dict):
            for key in ("text", "plain_text", "body"):
                value = content.get(key)
                if isinstance(value, str):
                    return value.strip()

        return ""

    def _extract_text_from_content(self, content: str) -> str:
        stripped = content.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(parsed, dict):
            for key in ("text", "content", "title"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return value.strip()
        if isinstance(parsed, str):
            return parsed.strip()
        return stripped

    def _extract_sender_id(self, item: Dict[str, Any]) -> Optional[str]:
        sender = item.get("sender")
        if isinstance(sender, dict):
            nested = self._first_non_empty(
                sender.get("sender_id"),
                sender.get("id"),
                sender.get("open_id"),
                sender.get("user_id"),
            )
            if nested:
                return nested
        return self._first_non_empty(item.get("sender_id"), item.get("senderId"), item.get("from"))

    def _extract_message_id(self, payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        direct = self._first_non_empty(payload.get("message_id"), payload.get("messageId"), payload.get("id"))
        if direct:
            return direct
        data = payload.get("data")
        if isinstance(data, dict):
            return self._first_non_empty(data.get("message_id"), data.get("messageId"), data.get("id"))
        return None

    @staticmethod
    def _first_non_empty(*values: Any) -> Optional[str]:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


class NativeLarkGatewayTransport(MessageTransport):
    """Transport backed by the local project-native Lark gateway."""

    def __init__(
        self,
        target: str,
        account_id: str,
        gateway_url: str = "http://127.0.0.1:18889",
        poll_interval: float = 1.0,
        history_limit: int = 20,
        verbose: bool = True,
    ):
        if not target:
            raise ValueError("Native Lark transport requires a target chat_id")
        if not account_id:
            raise ValueError("Native Lark transport requires an account_id")

        self.target = _normalize_chat_target(target)
        self.account_id = account_id
        self.gateway_url = gateway_url.rstrip("/")
        self.poll_interval = max(0.2, poll_interval)
        self.history_limit = max(1, history_limit)
        self.verbose = verbose
        self.agent_sender_map = _load_agent_sender_map()
        self.bot_aliases = _load_bot_aliases(account_id)
        self.agent_mention_map = _load_agent_mentions()
        self.agent_display_map = _load_agent_display_map()

        self._incoming: "queue.Queue[TransportMessage]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_seen_message_id: Optional[str] = None

    def start(self):
        if self._running:
            return
        self._prime_last_seen_message_id()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(
                f"🌐 Native Lark transport started "
                f"(account={self.account_id}, target={self.target}, gateway={self.gateway_url})"
            )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.verbose:
            print("🛑 Native Lark transport stopped")

    def get_command(self) -> Optional[str]:
        message = self._get_next_message()
        return message.text if message else None

    def get_response(self) -> Optional[str]:
        message = self._get_next_message()
        return message.text if message else None

    def send_message(
        self,
        message: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransportSendResult:
        outbound_message = self._format_outbound_message(message, recipient=recipient)
        response = requests.post(
            f"{self.gateway_url}/api/messages/send",
            json={
                "account_id": self.account_id,
                "chat_id": self.target,
                "message": outbound_message,
                "receive_id_type": "chat_id",
            },
            timeout=30,
        )
        if response.status_code != 200:
            return TransportSendResult(
                success=False,
                feedback="Native Lark gateway send failed",
                error=response.text,
            )
        payload = response.json()
        return TransportSendResult(
            success=True,
            feedback="Message sent via native Lark gateway",
            message_id=payload.get("message_id"),
            raw=payload,
        )

    def can_send_messages(self) -> bool:
        return True

    def should_mirror_local_speech(self) -> bool:
        return True

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_once()
            except Exception as error:  # pragma: no cover
                if self.verbose:
                    print(f"⚠️  Native Lark transport poll error: {error}")
            time.sleep(self.poll_interval)

    def _poll_once(self):
        params = {
            "account_id": self.account_id,
            "chat_id": self.target,
            "limit": self.history_limit,
        }
        if self._last_seen_message_id:
            params["after_id"] = self._last_seen_message_id
        response = requests.get(
            f"{self.gateway_url}/api/messages/read",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("messages", []):
            message_id = item.get("message_id")
            if message_id:
                self._last_seen_message_id = message_id
            if not self._should_accept_message(item):
                continue
            text = self._normalize_incoming_text(item)
            if not text:
                continue
            self._incoming.put(
                TransportMessage(
                    text=text,
                    message_id=message_id,
                    sender_id=item.get("sender_id"),
                    sender_type=item.get("sender_type"),
                    raw=item,
                )
            )

    def _prime_last_seen_message_id(self):
        response = requests.get(
            f"{self.gateway_url}/api/messages/read",
            params={
                "account_id": self.account_id,
                "chat_id": self.target,
                "limit": 1,
            },
            timeout=30,
        )
        if response.status_code != 200:
            return
        payload = response.json()
        messages = payload.get("messages", [])
        if messages:
            self._last_seen_message_id = messages[-1].get("message_id")

    def _get_next_message(self) -> Optional[TransportMessage]:
        try:
            return self._incoming.get_nowait()
        except queue.Empty:
            return None

    def _normalize_incoming_text(self, item: Dict[str, Any]) -> str:
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            return ""

        without_mentions = _strip_leading_mentions(raw_text)
        tagged_sender, tagged_recipient, cleaned_text = _extract_agent_envelope(without_mentions)
        sender_id = (item.get("sender_id") or "").strip()
        sender_name = (item.get("sender_name") or "").strip()
        sender_type = (item.get("sender_type") or "").strip().lower()

        speaker = self._resolve_agent_name(tagged_sender) or self.agent_sender_map.get(sender_id)
        if not speaker and sender_name:
            lowered_name = self._resolve_agent_name(sender_name) or sender_name.lower()
            if lowered_name in {"ur5e", "g1", "humanoid", "humanoid_robot"}:
                speaker = lowered_name
        if not speaker and sender_type == "app":
            speaker = "agent"

        if not speaker:
            speaker = "human"

        if not cleaned_text:
            return f"{speaker}:"
        return f"{speaker}: {cleaned_text}"

    def _should_accept_message(self, item: Dict[str, Any]) -> bool:
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            return False

        chat_type = str(item.get("chat_type") or "").strip().lower()
        if chat_type in {"p2p", "private"}:
            return True

        if not self.bot_aliases:
            return True

        alias_keys = {_canonicalize_alias(alias) for alias in self.bot_aliases if alias}
        mentions = _extract_mentions(raw_text) + _extract_structured_mentions(item)
        if any(_canonicalize_alias(mention) in alias_keys for mention in mentions if mention):
            return True

        without_mentions = _strip_leading_mentions(raw_text)
        _, tagged_recipient, _ = _extract_agent_envelope(without_mentions)
        resolved_recipient = self._resolve_agent_name(tagged_recipient)
        if resolved_recipient and _canonicalize_alias(resolved_recipient) in alias_keys:
            return True
        return False

    def _format_outbound_message(self, message: str, recipient: Optional[str] = None) -> str:
        stripped = str(message or "").strip()
        if not stripped:
            return ""
        if not recipient:
            return stripped

        recipient_key = recipient.strip()
        mention = self.agent_mention_map.get(recipient_key, recipient_key)
        mention = mention.lstrip("@").strip()
        if not mention:
            return stripped
        tagged_text = _apply_agent_envelope(
            stripped,
            sender=self.agent_display_map.get(self.account_id, self.account_id),
            recipient=self.agent_display_map.get(recipient_key, recipient_key),
        )
        if tagged_text.startswith(f"@{mention}"):
            return tagged_text
        return f"@{mention} {tagged_text}"

    def _resolve_agent_name(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        canonical = _canonicalize_alias(value)
        if not canonical:
            return None

        for agent_id, display_name in self.agent_display_map.items():
            if canonical in {
                _canonicalize_alias(agent_id),
                _canonicalize_alias(display_name),
            }:
                return agent_id

        lowered = str(value).strip().lower()
        if lowered in {"ur5e", "g1", "humanoid", "humanoid_robot"}:
            return lowered
        return lowered


def _normalize_chat_target(target: str) -> str:
    normalized = target.strip()
    for prefix in ("group:", "chat:", "channel:"):
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def _strip_leading_mentions(text: str) -> str:
    cleaned = text.strip()
    while cleaned.startswith("@"):
        cleaned = re.sub(r"^@\S+\s*", "", cleaned).strip()
    return cleaned


def _load_agent_sender_map() -> Dict[str, str]:
    raw = os.getenv("LARK_AGENT_SENDER_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                str(sender_id).strip(): str(agent_name).strip()
                for sender_id, agent_name in parsed.items()
                if str(sender_id).strip() and str(agent_name).strip()
            }
    except json.JSONDecodeError:
        pass

    mapping: Dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        sender_id, agent_name = item.split(":", 1)
        sender_id = sender_id.strip()
        agent_name = agent_name.strip()
        if sender_id and agent_name:
            mapping[sender_id] = agent_name
    return mapping


def _extract_mentions(text: str) -> List[str]:
    return [match.strip().lstrip("@").lower() for match in re.findall(r"@([^\s@]+)", text or "")]


def _extract_structured_mentions(item: Dict[str, Any]) -> List[str]:
    raw = item.get("raw_event")
    if not isinstance(raw, dict):
        return []

    message = raw.get("message")
    if not isinstance(message, dict):
        return []

    mentions = message.get("mentions")
    if not isinstance(mentions, list):
        return []

    extracted: List[str] = []
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        mention_id = mention.get("id")
        if not isinstance(mention_id, dict):
            mention_id = {}
        for candidate in (mention.get("name"), mention.get("key"), mention_id.get("open_id")):
            if isinstance(candidate, str) and candidate.strip():
                extracted.append(candidate.strip().lstrip("@").lower())
    return extracted


def _canonicalize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _extract_agent_envelope(text: str) -> tuple[Optional[str], Optional[str], str]:
    stripped = str(text or "").strip()
    sender: Optional[str] = None
    recipient: Optional[str] = None

    while True:
        match = re.match(r"^\[(agent|to):([A-Za-z0-9_\-]+)\]\s*(.*)$", stripped)
        if not match:
            break
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        stripped = match.group(3).strip()
        if key == "agent" and not sender:
            sender = value
        elif key == "to" and not recipient:
            recipient = value

    return sender, recipient, stripped


def _apply_agent_envelope(message: str, sender: str, recipient: Optional[str] = None) -> str:
    tagged_sender, tagged_recipient, cleaned_text = _extract_agent_envelope(message)
    final_sender = tagged_sender or sender.strip()
    final_recipient = tagged_recipient or (recipient.strip() if recipient else None)

    parts = []
    if final_sender:
        parts.append(f"[agent:{final_sender}]")
    if final_recipient:
        parts.append(f"[to:{final_recipient}]")
    if cleaned_text:
        parts.append(cleaned_text)
    return " ".join(parts).strip()


def _load_bot_aliases(account_id: str) -> set[str]:
    aliases = {account_id.strip().lower()} if account_id else set()
    defaults = {
        "g1": {"g1", "humanoid", "unitree-g1", "unitree_g1", "unitreeg1"},
        "ur5e": {"ur5e", "arm"},
    }
    aliases.update(defaults.get(account_id.strip().lower(), set()))

    raw = os.getenv("LARK_BOT_ALIASES", "").strip()
    if not raw:
        return aliases

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            value = parsed.get(account_id) or parsed.get(account_id.lower())
            if isinstance(value, list):
                aliases.update(str(item).strip().lower() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                aliases.add(value.strip().lower())
            return aliases
        if isinstance(parsed, list):
            aliases.update(str(item).strip().lower() for item in parsed if str(item).strip())
            return aliases
    except json.JSONDecodeError:
        pass

    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized:
            aliases.add(normalized)
    return aliases


def _load_agent_mentions() -> Dict[str, str]:
    raw = os.getenv("LARK_AGENT_MENTION_MAP", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                str(agent_name).strip(): str(mention).strip().lstrip("@")
                for agent_name, mention in parsed.items()
                if str(agent_name).strip() and str(mention).strip()
            }
    except json.JSONDecodeError:
        pass

    mapping: Dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        agent_name, mention = item.split(":", 1)
        agent_name = agent_name.strip()
        mention = mention.strip().lstrip("@")
        if agent_name and mention:
            mapping[agent_name] = mention
    return mapping


def _load_agent_display_map() -> Dict[str, str]:
    raw = os.getenv("LARK_AGENT_DISPLAY_MAP", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                str(agent_name).strip(): str(display_name).strip()
                for agent_name, display_name in parsed.items()
                if str(agent_name).strip() and str(display_name).strip()
            }
    except json.JSONDecodeError:
        pass

    mapping: Dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        agent_name, display_name = item.split(":", 1)
        agent_name = agent_name.strip()
        display_name = display_name.strip()
        if agent_name and display_name:
            mapping[agent_name] = display_name
    return mapping


def create_message_transport(
    mode: str,
    verbose: bool = True,
    **kwargs: Any,
) -> Optional[MessageTransport]:
    normalized = (mode or "none").strip().lower()
    if normalized in {"none", "off", "disabled"}:
        return None
    if normalized in {"voice", "speech", "asr"}:
        return VoiceTransport(verbose=verbose, asr=kwargs.get("asr"))
    if normalized in {"lark", "feishu", "native_lark", "native_feishu"}:
        target = kwargs.get("target") or os.getenv("LARK_TARGET") or os.getenv("OPENCLAW_LARK_TARGET")
        account_id = kwargs.get("account_id") or os.getenv("LARK_ACCOUNT_ID") or os.getenv("OPENCLAW_LARK_ACCOUNT")
        gateway_url = kwargs.get("gateway_url") or os.getenv("LARK_GATEWAY_URL", "http://127.0.0.1:18889")
        poll_interval = float(kwargs.get("poll_interval") or os.getenv("LARK_POLL_INTERVAL", "1.0"))
        return NativeLarkGatewayTransport(
            target=target,
            account_id=account_id,
            gateway_url=gateway_url,
            poll_interval=poll_interval,
            verbose=verbose,
        )
    if normalized in {"openclaw_lark", "openclaw-feishu"}:
        target = kwargs.get("target") or os.getenv("OPENCLAW_LARK_TARGET")
        account_id = kwargs.get("account_id") or os.getenv("OPENCLAW_LARK_ACCOUNT")
        channel = kwargs.get("channel") or os.getenv("OPENCLAW_LARK_CHANNEL", "feishu")
        poll_interval = float(kwargs.get("poll_interval") or os.getenv("OPENCLAW_LARK_POLL_INTERVAL", "1.0"))
        binary = kwargs.get("binary") or os.getenv("OPENCLAW_BIN", "openclaw")
        return OpenClawLarkTransport(
            target=target,
            account_id=account_id,
            channel=channel,
            poll_interval=poll_interval,
            binary=binary,
            verbose=verbose,
        )
    raise ValueError(f"Unsupported transport mode: {mode}")
