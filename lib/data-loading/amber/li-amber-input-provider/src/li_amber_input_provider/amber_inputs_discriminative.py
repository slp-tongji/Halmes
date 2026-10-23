from typing import Sequence

from li_image_operating import InMemoryImage
from li_llm_messages import LlmMessage
from li_llm_messages import LlmUserMessage

from li_amber_input_provider.amber_inputs_base import AmberInputsBase



class AmberInputsDiscriminative(AmberInputsBase):
    def get(self, id: str) -> Sequence[LlmMessage]:
        image: InMemoryImage
        query: str
        image, query = self.get_image_and_query(id)
        return [
            LlmUserMessage((image, 
                # AMBER does not provide details on how to build the prompt. 
                # We believe it's necessary to let the model know it should answer Yes or No.
                f"{query}\nPlease answer yes or no."
            ))
        ]
    