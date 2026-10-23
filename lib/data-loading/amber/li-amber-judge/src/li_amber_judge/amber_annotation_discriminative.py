from typing import Literal, TypedDict


class AmberAnnotationDiscriminative(TypedDict):
    id: int
    type: Literal["discriminative-attribute-state", 
                  "discriminative-attribute-action",
                  "discriminative-attribute-number",
                  "discriminative-hallucination",
                  "discriminative-relation", 
                  "relation"]
    truth: Literal["yes", "no"]