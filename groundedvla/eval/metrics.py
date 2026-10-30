"""Per-episode record and aggregate statistics (Step A4)."""
from dataclasses import asdict, dataclass, field

import numpy as np

LIFT_HEIGHT_M = 0.03


@dataclass
class EpisodeResult:
    # identity
    task: str
    instruction: str
    variant: str
    checkpoint: str
    seed: int
    condition: str
    episode_idx: int
    # success
    success: bool = False
    correct_object: bool = False
    grasp_success: bool = False
    correct_destination: bool = False
    # accuracy / safety / efficiency
    final_pos_error_cm: float = float("nan")
    collisions: int = 0
    steps: int = 0
    sim_time_s: float = 0.0
    latency_ms: list = field(default_factory=list)
    grounding_ms: list = field(default_factory=list)
    # diagnosis
    failure: str | None = None
    video: str | None = None

    def to_dict(self):
        d = asdict(self)
        for k in ("latency_ms", "grounding_ms"):
            v = np.asarray(d.pop(k))
            d[f"{k}_median"] = float(np.median(v)) if v.size else None
            d[f"{k}_p95"] = float(np.percentile(v, 95)) if v.size else None
        return d


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def horizontal_error_cm(obj_pos: np.ndarray, dest_pos: np.ndarray) -> float:
    return float(np.linalg.norm(obj_pos[:2] - dest_pos[:2]) * 100)
