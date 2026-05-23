import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.long_horizon_orchestration.run_eval import (
    build_record,
    build_scripted_trace,
    load_cases,
    summarize,
    write_results,
)
from experiments.long_horizon_orchestration.score import apply_physical_execution, score_trace


def _cases():
    return load_cases(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "long_horizon_orchestration"
        / "cases.jsonl"
    )


def test_long_horizon_suite_has_40_cases_split_by_family():
    cases = _cases()

    assert len(cases) == 40
    assert sum(1 for case in cases if case["family"] == "A" and case["platform"] == "ur5e") == 20
    assert sum(1 for case in cases if case["family"] == "B" and case["platform"] == "humanoid_g1") == 20


def test_omniact_scripted_action_correct_but_e2e_not_forced_to_one():
    case = _cases()[0]
    trace = build_scripted_trace(case, "omniact")
    failed_physical_trace = apply_physical_execution(trace, success_rate=0.0, seed=7)

    score = score_trace(case, failed_physical_trace)

    assert score["submetrics"]["action_correctness"] is True
    assert score["submetrics"]["physical_command_success"] is True
    assert score["submetrics"]["physical_execution_success"] is False
    assert score["success"] is False


def test_physical_sampling_is_seed_reproducible():
    case = _cases()[0]
    trace = build_scripted_trace(case, "omniact")

    first = apply_physical_execution(trace, success_rate=0.7, seed=11)
    second = apply_physical_execution(trace, success_rate=0.7, seed=11)

    assert first == second


def test_reactive_vla_loses_cyber_iot_but_keeps_physical_commands_when_present():
    case = _cases()[0]
    trace = apply_physical_execution(build_scripted_trace(case, "react_vla"), success_rate=1.0, seed=7)

    score = score_trace(case, trace)

    assert score["submetrics"]["cyber_iot_success"] is False
    assert score["submetrics"]["physical_command_success"] is True
    assert score["submetrics"]["action_correctness"] is False


def test_react_plus_vla_fails_async_visual_verification_cases():
    case = next(case for case in _cases() if case["requires_async_visual_verification"])
    trace = apply_physical_execution(build_scripted_trace(case, "react_plus_vla"), success_rate=1.0, seed=7)

    score = score_trace(case, trace)

    assert score["submetrics"]["vision_success"] is False
    assert score["success"] is False


def test_trace_dir_style_wrong_omniact_action_is_scored_as_failure():
    case = _cases()[0]
    trace = build_scripted_trace(case, "omniact")
    for step in trace["execution_history"]:
        if step["action_type"] == "act":
            step["parameters"]["item_name"] = "wrong_object"
            break
    trace = apply_physical_execution(trace, success_rate=1.0, seed=7)

    score = score_trace(case, trace)

    assert score["submetrics"]["physical_command_success"] is False
    assert score["success"] is False


def test_summary_and_csv_include_table1_metrics(tmp_path):
    cases = _cases()[:3]
    records = []
    for method in ("omniact", "react_vla", "react_plus_vla"):
        for case in cases:
            trace = apply_physical_execution(build_scripted_trace(case, method), success_rate=0.7, seed=7)
            records.append(build_record(case, trace))

    summary = summarize(records)
    write_results(tmp_path, records)

    assert {row["method"] for row in summary} == {"omniact", "react_vla", "react_plus_vla"}
    with (tmp_path / "table1_summary.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "overall_e2e_success" in rows[0]
    assert "cyber_iot_success" in rows[0]
    assert "physical_command_success" in rows[0]
    assert "physical_execution_success" in rows[0]
