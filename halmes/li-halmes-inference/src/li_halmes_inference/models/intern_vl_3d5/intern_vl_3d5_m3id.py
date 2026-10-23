from typing import Any

from li_halmes_configurations import HalmesTargetLlm
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig
from statictorch import Tensor2d

from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_base import InternVl3d5Base
from li_halmes_inference.models.samplers.m3id_sampler import M3idSampler
from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class InternVl3d5M3id(InternVl3d5Base):
    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, gate: SupportedGateProvider[SupportedGate]) -> None:
        super().__init__(variant_id, target, device, None, None)
        self.__sampler = M3idSampler()
        self.__gate = gate

    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        parameter: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        parameter_m3id: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        
        parameter_m3id["pixel_values"] = self.__sampler.modify_image(parameter_m3id["pixel_values"], None)
        
        model_kwargs: dict = {**parameter}
        model_kwargs_m3id: dict = {**parameter_m3id}
        shared: dict[str, Any] = {}
        key: str
        for key in ["input_ids", "logits_processor", "stopping_criteria", "generation_config", "synced_gpus"]:
            shared[key] = model_kwargs.pop(key)
            model_kwargs_m3id.pop(key)

        return self.__sampler.sample(
            self._model, 
            **shared, 
            streamer=None, 
            random=None, 
            model_kwargs=model_kwargs, 
            model_kwargs_m3id=model_kwargs_m3id,
            gate=self.__gate.new(mm_token_type_ids)
        )
