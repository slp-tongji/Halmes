from typing import Protocol, TypeVar

from statictorch import Tensor2d

from li_halmes_inference.supported_gate import SupportedGate


_T = TypeVar("_T", bound=SupportedGate, covariant=True)    # noqa: WVarDec01

class SupportedGateProvider(Protocol[_T]):
    def new(self, mm_token_type_ids: Tensor2d) -> _T:
        ...
