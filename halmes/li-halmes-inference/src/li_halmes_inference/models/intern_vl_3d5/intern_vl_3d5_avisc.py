from typing import Any
import typing

from li_halmes_configurations import HalmesTargetLlm
from statictorch import Tensor2d, Tensor3d
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

import transformers.utils.generic
from transformers.models.internvl.modeling_internvl import InternVLModelOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_base import InternVl3d5Base
from li_halmes_inference.models.samplers.avisc_sampler import AviscSampler
from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class InternVl3d5Avisc(InternVl3d5Base):
    class _Model(transformers.InternVLModel):
        def __init__(self, config: transformers.InternVLConfig) -> None:
            super().__init__(config)

        @transformers.utils.generic.can_return_tuple
        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            pixel_values: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: transformers.Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            vision_feature_layer: int | list[int] | list[int] | None = None,
            vision_feature_select_strategy: str | None = None,
            **kwargs: Any,
        ) -> tuple | InternVLModelOutputWithPast:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None

            image_features: torch.FloatTensor | None = None
            if pixel_values is not None:
                image_features = getattr(self, "get_image_features")(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    return_dict=True,
                ).pooler_output
                assert image_features is not None
                image_features = typing.cast(torch.FloatTensor,
                                             image_features.to(inputs_embeds.device, inputs_embeds.dtype))
                
                assert input_ids is not None
                special_image_mask: torch.Tensor = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = typing.cast(torch.FloatTensor,
                                            inputs_embeds.masked_scatter(special_image_mask, image_features))
                
                avisc: AviscSampler.Parameter | None = kwargs.get("avisc")
                if avisc is not None:
                    inputs_embeds = typing.cast(
                        torch.FloatTensor, avisc["mask"].mask_inputs(Tensor3d(inputs_embeds)))

            outputs: Any = getattr(self, "language_model")(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            return InternVLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

    class _ModelForConditionalGeneration(transformers.InternVLForConditionalGeneration):
        def __init__(self, config: transformers.InternVLConfig) -> None:
            super(transformers.InternVLForConditionalGeneration, self).__init__(config)

            self.model = InternVl3d5Avisc._Model(config)
            assert isinstance(config.text_config, Qwen3Config)
            self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
            self.post_init()


    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, gate: SupportedGateProvider[SupportedGate]) -> None:
        model: InternVl3d5Avisc._ModelForConditionalGeneration 
        model = InternVl3d5Avisc._ModelForConditionalGeneration.from_pretrained(
            target["model_path"],
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="eager"
        )
        super().__init__(variant_id, target, device, model, None)
        self.__sampler = AviscSampler()
        self.__gate = gate

    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        parameter: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        parameter_m3id: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        
        model_kwargs: dict = {**parameter}
        model_kwargs_avisc: dict = {**parameter_m3id}
        shared: dict[str, Any] = {}
        key: str
        for key in ["input_ids", "logits_processor", "stopping_criteria", "generation_config", "synced_gpus"]:
            shared[key] = model_kwargs.pop(key)
            model_kwargs_avisc.pop(key)

        return self.__sampler.sample(
            self._model, 
            **shared, 
            streamer=None, 
            random=None, 
            model_kwargs=model_kwargs, 
            model_kwargs_avisc=model_kwargs_avisc,
            gate=self.__gate.new(mm_token_type_ids),
            mm_token_type_ids=mm_token_type_ids
        )
