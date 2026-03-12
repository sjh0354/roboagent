#!/usr/bin/env bash

# Copy to env/planner.ur5e.env.sh and adjust for your UR5e bot.
export INTERACTION_TRANSPORT="lark"
export LARK_ACCOUNT_ID="ur5e"
export LARK_TARGET="group:oc_your_ur5e_chat_id"
export LARK_GATEWAY_URL="http://127.0.0.1:18889"

# Incoming sender normalization and group-chat mention behavior.
export LARK_AGENT_SENDER_MAP='{"ou_g1_sender_id":"g1"}'
export LARK_BOT_ALIASES='{"g1":["g1","humanoid","Unitree-G1"],"ur5e":["ur5e","arm","UR5E-Robot-arm"]}'
export LARK_AGENT_MENTION_MAP='{"ur5e":"UR5E-Robot-arm","g1":"Unitree-G1"}'
export LARK_AGENT_DISPLAY_MAP='{"g1":"Unitree-G1","ur5e":"UR5E-Robot-arm"}'
