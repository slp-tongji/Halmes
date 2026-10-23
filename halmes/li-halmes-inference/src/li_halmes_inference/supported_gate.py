from typing import Protocol, Sequence

from statictorch import Tensor4d


class SupportedGate(Protocol):
    def next(self, attentions: Sequence[Tensor4d]) -> bool:
        """
        Returns `True` if require intervention.
        """
        ...
