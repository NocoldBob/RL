from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "贪吃蛇"
sys.path.insert(0, str(SOURCE_DIR))
CONTINUOUS_DIR = Path(__file__).resolve().parents[1] / "连续控制"
sys.path.insert(0, str(CONTINUOUS_DIR))
LUNAR_DIR = Path(__file__).resolve().parents[1] / "月球着陆"
sys.path.insert(0, str(LUNAR_DIR))
MINIGRID_DIR = Path(__file__).resolve().parents[1] / "迷你网格"
sys.path.insert(0, str(MINIGRID_DIR))
VISUAL_DRIVING_DIR = Path(__file__).resolve().parents[1] / "视觉驾驶"
sys.path.insert(0, str(VISUAL_DRIVING_DIR))
MULTI_AGENT_DIR = Path(__file__).resolve().parents[1] / "多智能体"
sys.path.insert(0, str(MULTI_AGENT_DIR))
OFFLINE_RL_DIR = Path(__file__).resolve().parents[1] / "离线强化学习"
sys.path.insert(0, str(OFFLINE_RL_DIR))
MODEL_BASED_DIR = Path(__file__).resolve().parents[1] / "模型式强化学习"
sys.path.insert(0, str(MODEL_BASED_DIR))
