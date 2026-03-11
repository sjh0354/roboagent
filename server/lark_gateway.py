"""
Project-native Feishu/Lark gateway.

This server receives Feishu event callbacks and exposes a simple local API for
the planners to read/write robot messages without going through OpenClaw.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lark_client import LarkAccountConfig, LarkAccountRegistry, LarkClient


HOST = os.getenv("LARK_GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("LARK_GATEWAY_PORT", "18889"))
MESSAGE_BUFFER_SIZE = int(os.getenv("LARK_GATEWAY_BUFFER_SIZE", "200"))

app = FastAPI(title="RAS Lark Gateway")


@dataclass
class GatewayMessage:
    account_id: str
    chat_id: str
    chat_type: Optional[str]
    message_id: str
    sender_type: Optional[str]
    sender_id: Optional[str]
    sender_name: Optional[str]
    text: str
    create_time: Optional[str] = None
    raw_event: Optional[Dict] = None


class SendMessageRequest(BaseModel):
    account_id: str
    chat_id: str
    message: str
    receive_id_type: str = "chat_id"


class IngestMessageRequest(BaseModel):
    account_id: str
    chat_id: str
    chat_type: Optional[str] = None
    message_id: str
    text: str
    sender_type: Optional[str] = None
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    create_time: Optional[str] = None
    raw_event: Optional[Dict] = None


class LarkGatewayStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._messages: Dict[Tuple[str, str], Deque[GatewayMessage]] = defaultdict(
            lambda: deque(maxlen=MESSAGE_BUFFER_SIZE)
        )
        self._message_ids: set[str] = set()

    def add_message(self, message: GatewayMessage) -> bool:
        dedup_key = f"{message.account_id}:{message.message_id}"
        with self._lock:
            if dedup_key in self._message_ids:
                return False
            self._message_ids.add(dedup_key)
            self._messages[(message.account_id, message.chat_id)].append(message)
            return True

    def read_messages(
        self,
        account_id: str,
        chat_id: str,
        after_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[GatewayMessage]:
        with self._lock:
            items = list(self._messages.get((account_id, chat_id), []))
        if after_id:
            seen_after = False
            filtered: List[GatewayMessage] = []
            for item in items:
                if seen_after:
                    filtered.append(item)
                elif item.message_id == after_id:
                    seen_after = True
            items = filtered
        return items[-limit:]


class LarkGatewayRuntime:
    def __init__(self):
        self.accounts = LarkAccountRegistry.load_from_env()
        self.clients = {account_id: LarkClient(config) for account_id, config in self.accounts.items()}
        self.store = LarkGatewayStore()

    def get_account(self, account_id: str) -> LarkAccountConfig:
        account = self.accounts.get(account_id)
        if not account:
            raise KeyError(f"Unknown Lark account: {account_id}")
        return account

    def get_client(self, account_id: str) -> LarkClient:
        client = self.clients.get(account_id)
        if not client:
            raise KeyError(f"Unknown Lark client: {account_id}")
        return client


runtime = LarkGatewayRuntime()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "accounts": sorted(runtime.accounts.keys()),
        "time": time.time(),
    }


@app.get("/api/messages/read")
async def read_messages(
    account_id: str = Query(...),
    chat_id: str = Query(...),
    after_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
):
    try:
        runtime.get_account(account_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    items = runtime.store.read_messages(account_id=account_id, chat_id=chat_id, after_id=after_id, limit=limit)
    return {"messages": [asdict(item) for item in items]}


@app.post("/api/messages/send")
async def send_message(request: SendMessageRequest):
    try:
        client = runtime.get_client(request.account_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    try:
        payload = client.send_text_message(
            receive_id=request.chat_id,
            message=request.message,
            receive_id_type=request.receive_id_type,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    data = payload.get("data", {})
    return {
        "ok": True,
        "message_id": data.get("message_id"),
        "data": data,
    }


@app.post("/api/messages/ingest")
async def ingest_message(request: IngestMessageRequest):
    try:
        runtime.get_account(request.account_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    inserted = runtime.store.add_message(
        GatewayMessage(
            account_id=request.account_id,
            chat_id=request.chat_id,
            chat_type=request.chat_type,
            message_id=request.message_id,
            sender_type=request.sender_type,
            sender_id=request.sender_id,
            sender_name=request.sender_name,
            text=request.text,
            create_time=request.create_time,
            raw_event=request.raw_event,
        )
    )
    return {"ok": True, "inserted": inserted, "message_id": request.message_id}


@app.post("/webhook/{account_id}")
async def lark_webhook(account_id: str, request: Request):
    try:
        account = runtime.get_account(account_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    payload = await request.json()

    challenge = payload.get("challenge")
    if challenge:
        _verify_token(account, payload)
        return {"challenge": challenge}

    header = payload.get("header", {})
    event = payload.get("event", {})
    if not event and "schema" in payload and "event" in payload:
        event = payload["event"]

    _verify_token(account, payload)

    event_type = header.get("event_type") or payload.get("type")
    if event_type != "im.message.receive_v1":
        return {"ok": True, "ignored": True, "event_type": event_type}

    message = _parse_message_event(account_id, event)
    if message is None:
        return {"ok": True, "ignored": True}

    inserted = runtime.store.add_message(message)
    return {"ok": True, "inserted": inserted, "message_id": message.message_id}


def _verify_token(account: LarkAccountConfig, payload: Dict) -> None:
    expected = (account.verification_token or "").strip()
    if not expected:
        return

    header = payload.get("header", {})
    candidate = (
        header.get("token")
        or payload.get("token")
        or payload.get("verification_token")
    )
    if candidate != expected:
        raise HTTPException(status_code=403, detail="verification token mismatch")


def _parse_message_event(account_id: str, event: Dict) -> Optional[GatewayMessage]:
    sender = event.get("sender", {})
    sender_type = sender.get("sender_type")

    message = event.get("message", {})
    message_id = message.get("message_id")
    chat_id = message.get("chat_id")
    if not message_id or not chat_id:
        return None

    text = _extract_text(message.get("content", ""))
    if not text:
        return None

    sender_id = None
    sender_id_obj = sender.get("sender_id", {})
    if isinstance(sender_id_obj, dict):
        sender_id = sender_id_obj.get("open_id") or sender_id_obj.get("user_id") or sender_id_obj.get("union_id")

    sender_name = None
    sender_sender = sender.get("sender_id", {})
    if isinstance(sender_sender, dict):
        sender_name = sender_sender.get("name")
    if not sender_name:
        sender_name = sender.get("name")

    return GatewayMessage(
        account_id=account_id,
        chat_id=chat_id,
        chat_type=message.get("chat_type"),
        message_id=message_id,
        sender_type=sender_type,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        create_time=message.get("create_time"),
        raw_event=event,
    )


def _extract_text(content: str) -> str:
    if not content:
        return ""
    try:
        import json

        parsed = json.loads(content)
    except Exception:
        return str(content).strip()
    if isinstance(parsed, dict):
        text = parsed.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(parsed, str):
        return parsed.strip()
    return ""


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting RAS Lark Gateway on {HOST}:{PORT}")
    print(f"🤖 Accounts: {', '.join(sorted(runtime.accounts.keys())) or '(none configured)'}")
    uvicorn.run(app, host=HOST, port=PORT)
