# Remote Agent Messaging

- Use `send_agent_message` when communicating with another robot over a remote channel such as OpenClaw + Lark/Feishu.
- Prefer this action for inter-agent coordination that does not require physical co-location.
- Include a concise `message`.
- Include `recipient` when a specific peer or group target is required by the runtime.
- Do not use this action for human-facing speech in the same room. Use `speak` for local verbal interaction.
