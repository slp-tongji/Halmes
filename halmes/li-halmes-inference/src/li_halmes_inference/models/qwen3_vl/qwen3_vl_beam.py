from li_halmes_configurations import HalmesTargetLlm
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.qwen3_vl.qwen3_vl_base import Qwen3VlBase


class Qwen3VlBeam(Qwen3VlBase):
    def __init__(self, variant_id: str, target: HalmesTargetLlm, device: torch.device) -> None:
        super().__init__(variant_id, target, device, None, GenerationConfig(
            do_sample=False,
            num_beams=5
        ))

    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        return getattr(self._model, "generate")(
            **inputs,
            generation_config=generation_config
        )
    