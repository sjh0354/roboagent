"""
Experiment 4.4: API Dynamic Update Reflection Ablation

Compares:
- RoboClaw Full (reflection enabled): update skill doc after API drift error
- RoboClaw w/o Reflection: no adaptation, keeps retrying old payload

This experiment is simulation-only and does not require real hardware.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from executor.humanoid_executor import HumanoidExecutor
SKILL_PATH = REPO_ROOT / "agent" / "skills" / "common" / "query_weather" / "SKILL.md"
RESULT_DIR = REPO_ROOT / "experiments" / "results"

BASE_SKILL_CONTENT = """# Skill: Query Weather API

Use when:
- The user asks about weather/temperature/forecast in simulation experiments.
- You need structured weather data from the weather backend.

Action:
- `query_weather_api`

Current known payload format (legacy knowledge):
```json
{
  "city": "Beijing"
}
```

Rules:
- Use the known payload schema from this skill document.
- If backend reports missing required parameters, treat it as an API contract drift signal.
- In reflection-enabled mode, revise this skill doc and retry with corrected payload.
"""


@dataclass
class TrialRecord:
    mode: str
    trial_index: int
    success: bool
    attempts_used: int
    error_trace: List[str]
    updated_skill: bool


def reset_skill_doc() -> None:
    SKILL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKILL_PATH.write_text(BASE_SKILL_CONTENT, encoding="utf-8")


def read_skill_doc() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def extract_payload_from_skill(skill_text: str) -> Dict[str, Any]:
    # Try extracting first JSON code block
    blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", skill_text)
    if not blocks:
        return {"city": "Beijing"}

    for block in blocks:
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    return {"city": "Beijing"}


def reflective_update_skill_doc(error_message: str) -> bool:
    """Simple reflection policy: parse known missing-parameter errors and patch skill doc."""
    text = read_skill_doc()

    need_location = "missing required parameter: location_name" in (error_message or "")
    need_units = "missing required parameter: units" in (error_message or "")

    if not (need_location or need_units):
        return False

    new_payload = {
        "location_name": "Beijing",
        "units": "metric",
    }

    replacement_json = json.dumps(new_payload, ensure_ascii=False, indent=2)

    # Replace first JSON code block payload content
    updated = re.sub(
        r"```json\s*\{[\s\S]*?\}\s*```",
        f"```json\n{replacement_json}\n```",
        text,
        count=1,
    )

    # Update heading line for clarity
    updated = updated.replace(
        "Current known payload format (legacy knowledge):",
        "Current known payload format (updated after reflection):",
    )

    SKILL_PATH.write_text(updated, encoding="utf-8")
    return True


def run_single_trial(mode: str, max_attempts: int) -> TrialRecord:
    executor = HumanoidExecutor(simulation_mode=True, verbose=False)
    # API already changed in this experiment
    executor.weather_api.set_version("v2")

    error_trace: List[str] = []
    updated_skill = False

    for attempt in range(1, max_attempts + 1):
        skill_text = read_skill_doc()
        payload = extract_payload_from_skill(skill_text)
        result = executor.execute_action("tool", "query_weather_api", payload)

        if result.success:
            return TrialRecord(
                mode=mode,
                trial_index=-1,
                success=True,
                attempts_used=attempt,
                error_trace=error_trace,
                updated_skill=updated_skill,
            )

        err = result.error or result.data.get("error_message") or "unknown_error"
        error_trace.append(str(err))

        if mode == "full":
            changed = reflective_update_skill_doc(str(err))
            updated_skill = updated_skill or changed
        # mode == "no_reflection": do nothing and retry same old schema

    return TrialRecord(
        mode=mode,
        trial_index=-1,
        success=False,
        attempts_used=max_attempts,
        error_trace=error_trace,
        updated_skill=updated_skill,
    )


def cumulative_success_rate(records: List[TrialRecord]) -> List[float]:
    rates: List[float] = []
    success_count = 0
    for idx, rec in enumerate(records, start=1):
        if rec.success:
            success_count += 1
        rates.append(success_count / idx)
    return rates


def plot_ascii_curve(full_rates: List[float], no_rates: List[float]) -> str:
    lines = []
    lines.append("Trials vs Success Rate (ASCII)")
    lines.append("trial | full(reflection) | w/o_reflection")
    lines.append("------|-------------------|----------------")
    for i, (fr, nr) in enumerate(zip(full_rates, no_rates), start=1):
        lines.append(f"{i:>5} | {fr:>17.3f} | {nr:>14.3f}")
    return "\n".join(lines)


def summarize(records: List[TrialRecord]) -> Dict[str, Any]:
    total = len(records)
    success = sum(1 for r in records if r.success)
    attempts = [r.attempts_used for r in records]
    first_success_trial = next((i + 1 for i, r in enumerate(records) if r.success), None)

    return {
        "total_trials": total,
        "success_count": success,
        "success_rate": success / total if total else 0.0,
        "mean_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
        "first_success_trial": first_success_trial,
    }


def run_experiment(trials: int, max_attempts: int, seed: int) -> Dict[str, Any]:
    random.seed(seed)

    full_records: List[TrialRecord] = []
    no_records: List[TrialRecord] = []

    for t in range(1, trials + 1):
        # Full mode
        reset_skill_doc()
        full = run_single_trial(mode="full", max_attempts=max_attempts)
        full.trial_index = t
        full_records.append(full)

        # No-reflection mode
        reset_skill_doc()
        no_ref = run_single_trial(mode="no_reflection", max_attempts=max_attempts)
        no_ref.trial_index = t
        no_records.append(no_ref)

    full_rates = cumulative_success_rate(full_records)
    no_rates = cumulative_success_rate(no_records)

    return {
        "config": {
            "trials": trials,
            "max_attempts": max_attempts,
            "seed": seed,
            "api_drift": {
                "old": {"city": "Beijing"},
                "new": {"location_name": "Beijing", "units": "metric"},
            },
        },
        "summary": {
            "full": summarize(full_records),
            "no_reflection": summarize(no_records),
        },
        "rates": {
            "full": full_rates,
            "no_reflection": no_rates,
        },
        "records": {
            "full": [asdict(r) for r in full_records],
            "no_reflection": [asdict(r) for r in no_records],
        },
        "ascii_plot": plot_ascii_curve(full_rates, no_rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API dynamic-update reflection ablation experiment")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_experiment(trials=args.trials, max_attempts=args.max_attempts, seed=args.seed)

    out_json = RESULT_DIR / "api_reflection_ablation.json"
    out_txt = RESULT_DIR / "api_reflection_ablation.txt"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text(result["ascii_plot"] + "\n", encoding="utf-8")

    print("✅ Experiment completed")
    print(f"   JSON: {out_json}")
    print(f"   TXT:  {out_txt}")
    print()
    print(result["ascii_plot"])
    print()
    print("Summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
