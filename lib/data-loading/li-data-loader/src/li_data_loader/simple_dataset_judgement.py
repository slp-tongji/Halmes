import dataclasses


@dataclasses.dataclass(frozen=True)
class SimpleDatasetJudgement:
    judgement: bool
    judge_reason: str
