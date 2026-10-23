from typing import Literal, Sequence, TypedDict


class AmberAnnotationGenerative(TypedDict):
    id: int
    type: Literal["generative"]
    truth: Sequence[str]
    hallu: Sequence[str]