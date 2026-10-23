from typing import Sequence

from statictorch import Tensor2d, Tensor4d, Tensor3d, Tensor1d
from li_halmes_model import HalmesModel, HalmesGateParameter

from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class HalmesGate(SupportedGateProvider[SupportedGate]):
    class _Gate(SupportedGate):
        def __init__(self, gate: HalmesModel.HalmesGate) -> None:
            super().__init__()
            self.__gate = gate

        def next(self, attentions: Sequence[Tensor4d]) -> bool:
            assert attentions[0].size(0) == 1
            
            return self.__gate.next([
                Tensor3d(a.squeeze(0)) for a in attentions
            ])
    
    def __init__(self, model: HalmesModel, features: HalmesModel.HalmesFeatureSelector, parameter: HalmesGateParameter) -> None:
        super().__init__()
        self.__model = model
        self.__features = features
        self.__parameter = parameter

    def new(self, mm_token_type_ids: Tensor2d) -> _Gate:
        assert mm_token_type_ids.size(0) == 1

        return HalmesGate._Gate(self.__model.new_gate(
            self.__features, Tensor1d(mm_token_type_ids.squeeze(0)), self.__parameter)
        )
    