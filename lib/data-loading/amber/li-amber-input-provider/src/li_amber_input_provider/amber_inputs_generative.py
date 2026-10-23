from typing import Sequence

from li_llm_messages import LlmMessage
from li_llm_messages import LlmUserMessage

from li_amber_input_provider.amber_inputs_base import AmberInputsBase


class AmberInputsGenerative(AmberInputsBase):
    def get(self, id: str) -> Sequence[LlmMessage]:
        return [LlmUserMessage(self.get_image_and_query(id))]
    