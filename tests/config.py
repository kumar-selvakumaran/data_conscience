"""
Configuration for the tests
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR / "src"))


def load_env_vars():
    from dotenv import load_dotenv
    dotenv_path = str(PROJECT_DIR.parent.resolve() / '.env')
    load_dotenv(dotenv_path=dotenv_path)

load_env_vars()
