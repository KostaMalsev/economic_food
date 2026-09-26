"""Stable repository paths shared by analysis entry points."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data"
DEFAULT_SURVEY_DATA = DATA_DIR / "food_economics_2024.csv"
DEFAULT_REGRESSION_OUTPUT = REPOSITORY_ROOT / "2026-latet"
