"""
Mock weather backend used for simulation experiments.

Supports API version drift:
- v1 (legacy): payload requires {"city": "Beijing"}
- v2 (updated): payload requires {"location_name": "Beijing", "units": "metric"}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MockWeatherResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MockWeatherAPI:
    """In-process mock weather API with switchable contract versions."""

    def __init__(self, api_version: str = "v2"):
        if api_version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported api_version: {api_version}")
        self.api_version = api_version

    def set_version(self, api_version: str) -> None:
        if api_version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported api_version: {api_version}")
        self.api_version = api_version

    def query_weather(self, payload: Dict[str, Any]) -> MockWeatherResponse:
        if self.api_version == "v1":
            return self._query_v1(payload)
        return self._query_v2(payload)

    def _query_v1(self, payload: Dict[str, Any]) -> MockWeatherResponse:
        city = payload.get("city")
        if not city:
            return MockWeatherResponse(
                success=False,
                error_code="MISSING_REQUIRED_PARAMETER",
                error_message="missing required parameter: city",
            )

        return MockWeatherResponse(
            success=True,
            data={
                "location": city,
                "units": "metric",
                "temperature_c": 12,
                "condition": "Cloudy",
                "api_version": "v1",
            },
        )

    def _query_v2(self, payload: Dict[str, Any]) -> MockWeatherResponse:
        location_name = payload.get("location_name")
        units = payload.get("units")

        if not location_name:
            return MockWeatherResponse(
                success=False,
                error_code="MISSING_REQUIRED_PARAMETER",
                error_message="missing required parameter: location_name",
            )

        if units is None:
            return MockWeatherResponse(
                success=False,
                error_code="MISSING_REQUIRED_PARAMETER",
                error_message="missing required parameter: units",
            )

        if units != "metric":
            return MockWeatherResponse(
                success=False,
                error_code="INVALID_PARAMETER",
                error_message="invalid value for units: expected 'metric'",
            )

        return MockWeatherResponse(
            success=True,
            data={
                "location": location_name,
                "units": units,
                "temperature_c": 12,
                "condition": "Cloudy",
                "api_version": "v2",
            },
        )
