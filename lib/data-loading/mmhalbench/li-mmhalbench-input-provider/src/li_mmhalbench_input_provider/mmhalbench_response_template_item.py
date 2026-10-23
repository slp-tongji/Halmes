from typing import TypedDict


class MmhalbenchResponseTemplateItem(TypedDict):
    question_type: str
    question_topic: str
    image_id: str
    image_src: str
    image_content: list[str]
    question: str
    gt_answer: str
    model_answer: str
    