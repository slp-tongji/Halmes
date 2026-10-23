from typing import Any

from li_halmes_configurations import HalmesTargetLlm
from statictorch import Tensor2d
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_base import InternVl3d5Base
from li_halmes_inference.models.samplers.sid_sampler import SidSampler
from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class InternVl3d5Sid(InternVl3d5Base):
    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, gate: SupportedGateProvider[SupportedGate]) -> None:
        super().__init__(variant_id, target, device, None, None)
        self.__sampler = SidSampler(keep_ratio=0.1, agg_layer=2)
        self.__gate = gate

    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        parameter: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        
        parameter_sid: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        
        model_kwargs: dict = {**parameter}
        model_kwargs_sid: dict = {**parameter_sid}
        
        if "mm_token_type_ids" not in model_kwargs:
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids
            model_kwargs_sid["mm_token_type_ids"] = mm_token_type_ids
            
        shared: dict[str, Any] = {}
        key: str
        for key in ["input_ids", "logits_processor", "stopping_criteria", "generation_config", "synced_gpus"]:
            shared[key] = model_kwargs.pop(key)
            model_kwargs_sid.pop(key)

        return self.__sampler.sample(
            self._model, 
            **shared, 
            streamer=None, 
            random=None, 
            model_kwargs=model_kwargs, 
            model_kwargs_sid=model_kwargs_sid,
            gate=self.__gate.new(mm_token_type_ids),
            mm_token_type_ids=mm_token_type_ids
        )