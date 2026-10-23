from typing import NamedTuple


class HalmesGateParameter(NamedTuple):
    n: int = 15

    window: int = 15

    theta_high: float = 0.6
    theta_low: float = 0.5

    warm_up: int = 5
    cool_down: int = 5

    trend_window: int = 6
    trend_threshold: int = 0
    