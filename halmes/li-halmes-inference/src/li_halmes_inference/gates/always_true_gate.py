from typing import Sequence

from statictorch import Tensor2d, Tensor4d

from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class AlwaysTrueGate(SupportedGateProvider[SupportedGate]):
    class _Gate(SupportedGate):
        def next(self, attentions: Sequence[Tensor4d]) -> bool:
            return True
    
    def new(self, mm_token_type_ids: Tensor2d) -> _Gate:
        return AlwaysTrueGate._Gate()
    