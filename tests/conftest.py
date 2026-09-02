from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "贪吃蛇"
sys.path.insert(0, str(SOURCE_DIR))
CONTINUOUS_DIR = Path(__file__).resolve().parents[1] / "连续控制"
sys.path.insert(0, str(CONTINUOUS_DIR))
