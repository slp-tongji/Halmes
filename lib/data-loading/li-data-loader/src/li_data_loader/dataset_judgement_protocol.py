from typing import Protocol


class DatasetJudgementProtocol(Protocol):
    @property
    def judgement(self) -> bool:
        ...

    @property
    def judge_reason(self) -> str:
        ...
    