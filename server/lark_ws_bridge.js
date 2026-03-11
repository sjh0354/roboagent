"use strict";

const fs = require("fs");
const path = require("path");

function requireLarkSdk() {
  const candidates = [
    process.env.LARK_NODE_SDK_PATH,
    "@larksuiteoapi/node-sdk",
    "/opt/homebrew/lib/node_modules/openclaw/node_modules/@larksuiteoapi/node-sdk",
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (error) {
      // continue
    }
  }
  throw new Error(
    "Unable to load @larksuiteoapi/node-sdk. Set LARK_NODE_SDK_PATH or install the package."
  );
}

const Lark = requireLarkSdk();

const gatewayUrl = (process.env.LARK_GATEWAY_URL || "http://127.0.0.1:18889").replace(/\/+$/, "");
const domain = process.env.LARK_DOMAIN === "lark" ? Lark.Domain.Lark : Lark.Domain.Feishu;
const accountIds = (process.env.LARK_ACCOUNT_IDS || "")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);

function resolveAccounts() {
  if (accountIds.length === 0) {
    const appId = process.env.LARK_APP_ID;
    const appSecret = process.env.LARK_APP_SECRET;
    if (!appId || !appSecret) {
      return [];
    }
    return [
      {
        accountId: process.env.LARK_DEFAULT_ACCOUNT || "default",
        appId,
        appSecret,
        verificationToken: process.env.LARK_VERIFICATION_TOKEN || "",
        encryptKey: process.env.LARK_ENCRYPT_KEY || "",
      },
    ];
  }

  return accountIds
    .map((accountId) => {
      const prefix = accountId.toUpperCase().replace(/-/g, "_");
      const appId = process.env[`LARK_${prefix}_APP_ID`];
      const appSecret = process.env[`LARK_${prefix}_APP_SECRET`];
      if (!appId || !appSecret) {
        return null;
      }
      return {
        accountId,
        appId,
        appSecret,
        verificationToken: process.env[`LARK_${prefix}_VERIFICATION_TOKEN`] || "",
        encryptKey: process.env[`LARK_${prefix}_ENCRYPT_KEY`] || "",
      };
    })
    .filter(Boolean);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

function parseTextContent(content) {
  if (!content) {
    return "";
  }
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed.text === "string") {
      return parsed.text.trim();
    }
    if (typeof parsed === "string") {
      return parsed.trim();
    }
  } catch (error) {
    return String(content).trim();
  }
  return "";
}

async function ingestMessage(accountId, event) {
  const senderId =
    event.sender?.sender_id?.open_id ||
    event.sender?.sender_id?.user_id ||
    event.sender?.sender_id?.union_id ||
    null;

  const text = parseTextContent(event.message?.content);
  if (!text) {
    return;
  }

  await postJson(`${gatewayUrl}/api/messages/ingest`, {
    account_id: accountId,
    chat_id: event.message.chat_id,
    chat_type: event.message?.chat_type || null,
    message_id: event.message.message_id,
    text,
    sender_type: event.sender?.sender_type || null,
    sender_id: senderId,
    sender_name: event.sender?.sender_id?.name || null,
    create_time: event.message.create_time || null,
    raw_event: event,
  });
}

function startAccount(account) {
  const eventDispatcher = new Lark.EventDispatcher({
    encryptKey: account.encryptKey || undefined,
    verificationToken: account.verificationToken || undefined,
  }).register({
    "im.message.receive_v1": async (data) => {
      const event = data.event ? data.event : data;
      try {
        await ingestMessage(account.accountId, event);
        console.log(
          `ingested ${account.accountId} ${event?.message?.chat_id || "unknown"} ${event?.message?.message_id || "unknown"}`
        );
      } catch (error) {
        console.error(`ingest failed for ${account.accountId}:`, error);
      }
    },
  });

  const wsClient = new Lark.WSClient({
    appId: account.appId,
    appSecret: account.appSecret,
    domain,
    loggerLevel: Lark.LoggerLevel.info,
  });

  wsClient.start({ eventDispatcher });
  console.log(`started lark ws account=${account.accountId}`);
}

async function main() {
  const accounts = resolveAccounts();
  if (accounts.length === 0) {
    console.error("No Lark accounts configured");
    process.exit(1);
  }

  for (const account of accounts) {
    startAccount(account);
  }

  process.on("SIGINT", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
