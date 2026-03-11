#!/usr/bin/env bash

# Copy to env/planner.ur5e.env.sh and adjust for your UR5e bot.
export INTERACTION_TRANSPORT="lark"
export LARK_ACCOUNT_ID="ur5e"
export LARK_TARGET="group:oc_your_ur5e_chat_id"
export LARK_GATEWAY_URL="http://127.0.0.1:18889"

# Incoming sender normalization and group-chat mention behavior.
export LARK_AGENT_SENDER_MAP='{"ou_g1_sender_id":"g1"}'
export LARK_BOT_ALIASES='{"ur5e":["ur5e","arm"]}'
export LARK_AGENT_MENTION_MAP='{"ur5e":"ur5e","g1":"g1"}'
