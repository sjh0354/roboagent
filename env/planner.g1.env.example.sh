#!/usr/bin/env bash

# Copy to env/planner.g1.env.sh and adjust for your G1 bot.
export INTERACTION_TRANSPORT="lark"
export LARK_ACCOUNT_ID="g1"
export LARK_TARGET="group:oc_your_g1_chat_id"
export LARK_GATEWAY_URL="http://127.0.0.1:18889"

# Incoming sender normalization and group-chat mention behavior.
export LARK_AGENT_SENDER_MAP='{"ou_ur5e_sender_id":"ur5e"}'
export LARK_BOT_ALIASES='{"g1":["g1","humanoid"]}'
export LARK_AGENT_MENTION_MAP='{"ur5e":"ur5e","g1":"g1"}'
