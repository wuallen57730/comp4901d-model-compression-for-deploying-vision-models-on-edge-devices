"""Realtime edge demo backend package for Jetson-side inference."""

from .config import DemoConfig, parse_config

__all__ = ["DemoConfig", "parse_config"]
