from li_halmes_configurations import HalmesTargetLlm
from statictorch import Tensor2d
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_base import InternVl3d5Base


class InternVl3d5Vanilla(InternVl3d5Base):
    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device) -> None:
        super().__init__(variant_id, target, device, None, None)

    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        return getattr(self._model, "generate")(
            **inputs,
            generation_config=generation_config
        )