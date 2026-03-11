"""
Native Feishu/Lark HTTP client for this repository.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests


FEISHU_BASE_URL = os.getenv("LARK_BASE_URL", "https://open.feishu.cn").rstrip("/")


@dataclass
class LarkAccountConfig:
    account_id: str
    app_id: str
    app_secret: str
    verification_token: Optional[str] = None
    encrypt_key: Optional[str] = None
    default_chat_id: Optional[str] = None
    base_url: str = FEISHU_BASE_URL


class LarkAccountRegistry:
    """Loads multi-bot account config from environment variables."""

    @staticmethod
    def load_from_env() -> Dict[str, LarkAccountConfig]:
        accounts: Dict[str, LarkAccountConfig] = {}
        account_ids = [item.strip() for item in os.getenv("LARK_ACCOUNT_IDS", "").split(",") if item.strip()]

        if not account_ids:
            single_app_id = os.getenv("LARK_APP_ID")
            single_secret = os.getenv("LARK_APP_SECRET")
            if single_app_id and single_secret:
                account_id = os.getenv("LARK_DEFAULT_ACCOUNT", "default")
                accounts[account_id] = LarkAccountConfig(
                    account_id=account_id,
                    app_id=single_app_id,
                    app_secret=single_secret,
                    verification_token=os.getenv("LARK_VERIFICATION_TOKEN"),
                    encrypt_key=os.getenv("LARK_ENCRYPT_KEY"),
                    default_chat_id=os.getenv("LARK_DEFAULT_CHAT_ID"),
                )
            return accounts

        for account_id in account_ids:
            prefix = account_id.upper().replace("-", "_")
            app_id = os.getenv(f"LARK_{prefix}_APP_ID")
            app_secret = os.getenv(f"LARK_{prefix}_APP_SECRET")
            if not app_id or not app_secret:
                continue
            accounts[account_id] = LarkAccountConfig(
                account_id=account_id,
                app_id=app_id,
                app_secret=app_secret,
                verification_token=os.getenv(f"LARK_{prefix}_VERIFICATION_TOKEN"),
                encrypt_key=os.getenv(f"LARK_{prefix}_ENCRYPT_KEY"),
                default_chat_id=os.getenv(f"LARK_{prefix}_DEFAULT_CHAT_ID"),
                base_url=os.getenv(f"LARK_{prefix}_BASE_URL", FEISHU_BASE_URL).rstrip("/"),
            )
        return accounts


class LarkClient:
    def __init__(self, config: LarkAccountConfig, timeout: float = 30.0):
        self.config = config
        self.timeout = timeout
        self._token_lock = threading.Lock()
        self._tenant_access_token: Optional[str] = None
        self._token_expire_at = 0.0

    def _get_tenant_access_token(self) -> str:
        now = time.time()
        if self._tenant_access_token and now < self._token_expire_at:
            return self._tenant_access_token

        with self._token_lock:
            now = time.time()
            if self._tenant_access_token and now < self._token_expire_at:
                return self._tenant_access_token

            response = requests.post(
                f"{self.config.base_url}/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.config.app_id,
                    "app_secret": self.config.app_secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(payload.get("msg") or "failed to fetch tenant access token")

            token = payload.get("tenant_access_token")
            expire = int(payload.get("expire", 7200))
            if not token:
                raise RuntimeError("tenant_access_token missing in Feishu response")

            self._tenant_access_token = token
            self._token_expire_at = time.time() + max(60, expire - 120)
            return token

    def send_text_message(
        self,
        receive_id: str,
        message: str,
        receive_id_type: str = "chat_id",
    ) -> Dict:
        token = self._get_tenant_access_token()
        response = requests.post(
            f"{self.config.base_url}/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "receive_id": receive_id,
                "msg_type": "text",
                "content": f'{{"text":"{_escape_lark_text(message)}"}}',
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("msg") or "failed to send lark message")
        return payload


def _escape_lark_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )
