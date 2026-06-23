"""Competence-gated curriculum control for the GenAIRR gym."""
from .config import GateSpec, GymConfig, default_gates
from .state import GateStatus, AxisStat, GymState, composite_score
from .ladder import RankLadder
from .gate import PromotionGate
from .plateau import PlateauDetector
from .hud import GymHUD
from .controller import GymController

__all__ = [
    "GateSpec", "GymConfig", "default_gates", "GateStatus", "AxisStat", "GymState",
    "composite_score", "RankLadder", "PromotionGate", "PlateauDetector", "GymHUD",
    "GymController",
]
