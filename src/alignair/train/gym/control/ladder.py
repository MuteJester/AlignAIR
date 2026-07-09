"""RankLadder: integer floor -> scalar curriculum progress -> GenAIRR params."""
from ..curriculum import Curriculum

_ROOM_NAMES = (
    "Training Grounds", "Mutant Foothills", "SHM Caverns", "Trimmed Halls",
    "Indel Marsh", "Noisy Bazaar", "Fragment Ruins", "Heavy-SHM Tower",
    "Orientation Abyss", "The Gauntlet",
)


class RankLadder:
    def __init__(self, curriculum: Curriculum | None = None, n_levels: int = 10):
        self.curriculum = curriculum or Curriculum()
        self.n_levels = n_levels

    @property
    def top(self) -> int:
        return self.n_levels - 1

    def progress(self, level: int) -> float:
        level = max(0, min(self.top, level))
        return level / max(self.top, 1)

    def params(self, level: int) -> dict:
        return self.curriculum.params(self.progress(level))

    def name(self, level: int) -> str:
        if 0 <= level < len(_ROOM_NAMES):
            return _ROOM_NAMES[level]
        return f"Floor {level + 1}"
