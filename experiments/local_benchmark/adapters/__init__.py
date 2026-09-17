"""Agent/runtime adapters for the common simulator."""

from .hermes import run_hermes_episode
from .openclaw import run_openclaw_episode

__all__ = ["run_hermes_episode", "run_openclaw_episode"]
