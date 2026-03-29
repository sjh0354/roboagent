#!/usr/bin/env bash

# Copy to env/planner.g1.env.sh and adjust for your G1 bot.
export INTERACTION_TRANSPORT="lark"
export LARK_ACCOUNT_ID="g1"
export LARK_TARGET="group:oc_your_g1_chat_id"
export LARK_GATEWAY_URL="http://127.0.0.1:18889"

# Incoming sender normalization and group-chat mention behavior.
export LARK_AGENT_SENDER_MAP='{"ou_ur5e_sender_id":"ur5e"}'
export LARK_BOT_ALIASES='{"g1":["g1","humanoid","Unitree-G1"],"ur5e":["ur5e","arm","UR5E-Robot-arm"]}'
export LARK_AGENT_MENTION_MAP='{"ur5e":"UR5E-Robot-arm","g1":"Unitree-G1"}'
export LARK_AGENT_DISPLAY_MAP='{"g1":"Unitree-G1","ur5e":"UR5E-Robot-arm"}'

# Optional memory/runtime tuning.
# Conservative per-model context windows. Override per deployment if needed.
export PLANNER_CONTEXT_WINDOW_TOKENS_GEMINI_2_0_FLASH_EXP="128000"
export PLANNER_CONTEXT_WINDOW_TOKENS_GEMINI_2_0_FLASH_THINKING_EXP_01_21="128000"
export PLANNER_CONTEXT_WINDOW_TOKENS_GEMINI_3_PRO_PREVIEW="128000"
export PLANNER_CONTEXT_WINDOW_TOKENS_GEMINI_2_5_FLASH_LITE="128000"

# Hierarchical memory triggers and budget buckets.
export MEMORY_TRIGGER_RATIO="0.25"
export MEMORY_FORCE_SUMMARY_SECONDS="7200"
export MEMORY_CONTEXT_BUDGET_RATIO="0.35"
export MEMORY_OVERVIEW_BUDGET_RATIO="0.35"
export MEMORY_LONGTERM_BUDGET_RATIO="0.25"
export MEMORY_SHORTTERM_BUDGET_RATIO="0.20"
export MEMORY_RAW_BUDGET_RATIO="0.20"
export MEMORY_LONGTERM_PROMOTION_BATCH="3"
export MEMORY_OVERVIEW_PROMOTION_BATCH="3"
