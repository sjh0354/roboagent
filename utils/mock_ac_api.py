"""
Mock air-conditioner backend used for API-drift reflection experiments.

The current contract models a stricter AC control API with multiple optional
configuration fields. The experiment intentionally starts from stale API
knowledge so a single call can expose several issues in sequence.

When the payload is invalid, the API can emit either:
- informative feedback: precise missing/invalid parameter messages
- opaque feedback: generic request rejection without repair guidance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


VALID_ACTIONS = ("turn_on", "turn_off")
VALID_WINDS = ("low", "medium", "high")
VALID_MODES = ("cool", "heat", "auto", "fan_only")
VALID_SWINGS = ("off", "vertical", "horizontal", "both")
VALID_PRESETS = ("normal", "eco", "boost")
VALID_DISPLAYS = ("on", "off")
VALID_PURIFIERS = ("on", "off")
VALID_SLEEP_TIMERS = ("30m", "60m", "90m")
VALID_ENERGY_SAVERS = ("on", "off")
VALID_AIRFLOW_PATTERNS = ("direct", "breeze", "diffuse")
VALID_AMBIENCE_LIGHTS = ("warm", "neutral", "cool")
VALID_DEHUMIDIFY_LEVELS = ("low", "medium", "high")


@dataclass
class MockACResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ValidationIssue:
    error_code: str
    field: str
    informative_message: str


class MockAirConditionerAPI:
    """In-process AC API with configurable error feedback style."""

    def control(
        self,
        payload: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        feedback_style: str = "informative",
        opaque_message: Optional[str] = None,
    ) -> MockACResponse:
        issue = self._validate(payload, required_fields=required_fields)
        if issue is None:
            response_data: Dict[str, Any] = {
                "device": "air_conditioner",
                "action": payload["action"],
                "api_version": "v2",
            }
            if "temperature" in payload:
                response_data["temperature"] = payload["temperature"]
            if "wind" in payload:
                response_data["wind"] = payload["wind"]
            if "mode" in payload:
                response_data["mode"] = payload["mode"]
            if "swing" in payload:
                response_data["swing"] = payload["swing"]
            if "preset" in payload:
                response_data["preset"] = payload["preset"]
            if "humidity" in payload:
                response_data["humidity"] = payload["humidity"]
            if "purifier" in payload:
                response_data["purifier"] = payload["purifier"]
            if "display" in payload:
                response_data["display"] = payload["display"]
            if "sleep_timer" in payload:
                response_data["sleep_timer"] = payload["sleep_timer"]
            if "energy_saver" in payload:
                response_data["energy_saver"] = payload["energy_saver"]
            if "airflow_pattern" in payload:
                response_data["airflow_pattern"] = payload["airflow_pattern"]
            if "ambience_light" in payload:
                response_data["ambience_light"] = payload["ambience_light"]
            if "dehumidify_level" in payload:
                response_data["dehumidify_level"] = payload["dehumidify_level"]

            return MockACResponse(
                success=True,
                data=response_data,
            )

        if feedback_style == "opaque":
            return MockACResponse(
                success=False,
                error_code="BAD_REQUEST",
                error_message=opaque_message or "bad request",
            )

        return MockACResponse(
            success=False,
            error_code=issue.error_code,
            error_message=issue.informative_message,
        )

    def _validate(
        self,
        payload: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
    ) -> Optional[ValidationIssue]:
        required = list(
            required_fields
            or [
                "action",
                "temperature",
                "wind",
                "mode",
                "swing",
                "preset",
                "humidity",
                "purifier",
                "display",
                "sleep_timer",
                "energy_saver",
                "airflow_pattern",
                "ambience_light",
                "dehumidify_level",
            ]
        )

        action = payload.get("action")
        if "action" in required and action is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="action",
                informative_message="missing required parameter: action",
            )
        if action is not None and action not in VALID_ACTIONS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="action",
                informative_message=(
                    "invalid value for action: allowed values = "
                    f"[{', '.join(VALID_ACTIONS)}]"
                ),
            )

        temperature = payload.get("temperature")
        if "temperature" in required and temperature is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="temperature",
                informative_message="missing required parameter: temperature",
            )
        if temperature is not None and not isinstance(temperature, int):
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="temperature",
                informative_message="invalid value for temperature: expected integer in range [16, 30]",
            )
        if isinstance(temperature, int) and (temperature < 16 or temperature > 30):
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="temperature",
                informative_message="invalid value for temperature: expected integer in range [16, 30]",
            )

        wind = payload.get("wind")
        if "wind" in required and wind is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="wind",
                informative_message="missing required parameter: wind",
            )
        if wind is not None and wind not in VALID_WINDS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="wind",
                informative_message=(
                    "invalid value for wind: allowed values = "
                    f"[{', '.join(VALID_WINDS)}]"
                ),
            )

        mode = payload.get("mode")
        if "mode" in required and mode is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="mode",
                informative_message="missing required parameter: mode",
            )
        if mode is not None and mode not in VALID_MODES:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="mode",
                informative_message=(
                    "invalid value for mode: allowed values = "
                    f"[{', '.join(VALID_MODES)}]"
                ),
            )

        swing = payload.get("swing")
        if "swing" in required and swing is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="swing",
                informative_message="missing required parameter: swing",
            )
        if swing is not None and swing not in VALID_SWINGS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="swing",
                informative_message=(
                    "invalid value for swing: allowed values = "
                    f"[{', '.join(VALID_SWINGS)}]"
                ),
            )

        preset = payload.get("preset")
        if "preset" in required and preset is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="preset",
                informative_message="missing required parameter: preset",
            )
        if preset is not None and preset not in VALID_PRESETS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="preset",
                informative_message=(
                    "invalid value for preset: allowed values = "
                    f"[{', '.join(VALID_PRESETS)}]"
                ),
            )

        humidity = payload.get("humidity")
        if "humidity" in required and humidity is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="humidity",
                informative_message="missing required parameter: humidity",
            )
        if humidity is not None and not isinstance(humidity, int):
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="humidity",
                informative_message="invalid value for humidity: expected integer in range [35, 65]",
            )
        if isinstance(humidity, int) and (humidity < 35 or humidity > 65):
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="humidity",
                informative_message="invalid value for humidity: expected integer in range [35, 65]",
            )

        purifier = payload.get("purifier")
        if "purifier" in required and purifier is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="purifier",
                informative_message="missing required parameter: purifier",
            )
        if purifier is not None and purifier not in VALID_PURIFIERS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="purifier",
                informative_message=(
                    "invalid value for purifier: allowed values = "
                    f"[{', '.join(VALID_PURIFIERS)}]"
                ),
            )

        display = payload.get("display")
        if "display" in required and display is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="display",
                informative_message="missing required parameter: display",
            )
        if display is not None and display not in VALID_DISPLAYS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="display",
                informative_message=(
                    "invalid value for display: allowed values = "
                    f"[{', '.join(VALID_DISPLAYS)}]"
                ),
            )

        sleep_timer = payload.get("sleep_timer")
        if "sleep_timer" in required and sleep_timer is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="sleep_timer",
                informative_message="missing required parameter: sleep_timer",
            )
        if sleep_timer is not None and sleep_timer not in VALID_SLEEP_TIMERS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="sleep_timer",
                informative_message=(
                    "invalid value for sleep_timer: allowed values = "
                    f"[{', '.join(VALID_SLEEP_TIMERS)}]"
                ),
            )

        energy_saver = payload.get("energy_saver")
        if "energy_saver" in required and energy_saver is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="energy_saver",
                informative_message="missing required parameter: energy_saver",
            )
        if energy_saver is not None and energy_saver not in VALID_ENERGY_SAVERS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="energy_saver",
                informative_message=(
                    "invalid value for energy_saver: allowed values = "
                    f"[{', '.join(VALID_ENERGY_SAVERS)}]"
                ),
            )

        airflow_pattern = payload.get("airflow_pattern")
        if "airflow_pattern" in required and airflow_pattern is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="airflow_pattern",
                informative_message="missing required parameter: airflow_pattern",
            )
        if airflow_pattern is not None and airflow_pattern not in VALID_AIRFLOW_PATTERNS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="airflow_pattern",
                informative_message=(
                    "invalid value for airflow_pattern: allowed values = "
                    f"[{', '.join(VALID_AIRFLOW_PATTERNS)}]"
                ),
            )

        ambience_light = payload.get("ambience_light")
        if "ambience_light" in required and ambience_light is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="ambience_light",
                informative_message="missing required parameter: ambience_light",
            )
        if ambience_light is not None and ambience_light not in VALID_AMBIENCE_LIGHTS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="ambience_light",
                informative_message=(
                    "invalid value for ambience_light: allowed values = "
                    f"[{', '.join(VALID_AMBIENCE_LIGHTS)}]"
                ),
            )

        dehumidify_level = payload.get("dehumidify_level")
        if "dehumidify_level" in required and dehumidify_level is None:
            return ValidationIssue(
                error_code="MISSING_REQUIRED_PARAMETER",
                field="dehumidify_level",
                informative_message="missing required parameter: dehumidify_level",
            )
        if dehumidify_level is not None and dehumidify_level not in VALID_DEHUMIDIFY_LEVELS:
            return ValidationIssue(
                error_code="INVALID_PARAMETER",
                field="dehumidify_level",
                informative_message=(
                    "invalid value for dehumidify_level: allowed values = "
                    f"[{', '.join(VALID_DEHUMIDIFY_LEVELS)}]"
                ),
            )

        return None
