from typing import Any
import typing

from li_halmes_configurations import HalmesTargetLlm
from statictorch import Tensor2d, Tensor3d
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

import transformers.utils.generic
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from li_halmes_inference.models.qwen3_vl.qwen3_vl_base import Qwen3VlBase
from li_halmes_inference.models.samplers.avisc_sampler import AviscSampler
from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class Qwen3VlAvisc(Qwen3VlBase):
    class _Model(transformers.Qwen3VLModel):
        def __init__(self, config: transformers.Qwen3VLConfig) -> None:
            super().__init__(config)

        @transformers.utils.generic.can_return_tuple
        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: transformers.Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
            **kwargs: Any,
        ) -> tuple | Qwen3VLModelOutputWithPast:
            assert pixel_values_videos is None
            assert position_ids is not None

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None

            visual_pos_masks: torch.Tensor | None = None
            deepstack_visual_embeds: list[torch.Tensor] | None = None

            if pixel_values is not None:
                image_outputs: Any = getattr(self, "get_image_features")(
                    pixel_values, image_grid_thw, return_dict=True
                )
                image_embeds: torch.Tensor = torch.cat(image_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype)
                deepstack_image_embeds: list[torch.Tensor] = image_outputs.deepstack_features

                assert input_ids is not None
                image_mask: torch.Tensor
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, 
                    image_features=typing.cast(torch.FloatTensor, image_embeds)
                )
                
                inputs_embeds = typing.cast(torch.FloatTensor, inputs_embeds.masked_scatter(
                    typing.cast(torch.FloatTensor, image_mask), image_embeds))
                
                image_mask = image_mask[..., 0]
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds

                avisc: AviscSampler.Parameter | None = kwargs.get("avisc")
                if avisc is not None:
                    inputs_embeds = typing.cast(
                        torch.FloatTensor, avisc["mask"].mask_inputs(Tensor3d(inputs_embeds)))
                    deepstack_visual_embeds = typing.cast(
                        list[torch.Tensor], 
                        avisc["mask"].mask_deepstack(typing.cast(list[Tensor2d], deepstack_visual_embeds), 
                                                     Tensor2d(visual_pos_masks)))

            outputs: Any = getattr(self, "language_model")(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

            return Qwen3VLModelOutputWithPast(
                **outputs,
                rope_deltas=getattr(self, "rope_deltas"),
            )

    class _ModelForConditionalGeneration(transformers.Qwen3VLForConditionalGeneration):
        def __init__(self, config: transformers.Qwen3VLConfig) -> None:
            super(transformers.Qwen3VLForConditionalGeneration, self).__init__(config)

            self.model = Qwen3VlAvisc._Model(config)

            assert isinstance(config.text_config, Qwen3VLTextConfig)
            self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

            self.post_init()


    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, gate: SupportedGateProvider[SupportedGate]) -> None:
        model: Qwen3VlAvisc._ModelForConditionalGeneration 
        model = Qwen3VlAvisc._ModelForConditionalGeneration.from_pretrained(
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
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        parameter: Qwen3VlBase._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        parameter_m3id: Qwen3VlBase._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        
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
            gate=self.__gate.new(model_kwargs["mm_token_type_ids"]),
            mm_token_type_ids=model_kwargs["mm_token_type_ids"]
        )
