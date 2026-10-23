from typing import Any, Iterable, Sequence

from li_halmes_configurations import HalmesTargetLlm
from li_llm_messages import LlmAssistantMessage, LlmMessage
from statictorch import Tensor2d
import torch
import transformers

from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_base import InternVl3d5Base
from li_halmes_inference.models.samplers.selfaug_sampler import SelfaugSampler
from li_halmes_inference.supported_gate import SupportedGate
from li_halmes_inference.supported_gate_provider import SupportedGateProvider


class InternVl3d5Selfaug(InternVl3d5Base):
    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, gate: SupportedGateProvider[SupportedGate]) -> None:
        super().__init__(variant_id, target, device, None, None)
        self.__sampler = SelfaugSampler()
        self.__gate = gate

    def __chat_greedy(self, messages: Sequence[LlmMessage]) -> LlmAssistantMessage:
        inputs: object = self._processor.apply_chat_template(
            self._messages_to_transformers(messages),
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        assert isinstance(inputs, transformers.feature_extraction_utils.BatchFeature)
        inputs = inputs.to(self._device, dtype=torch.bfloat16)
        
        config: GenerationConfig = self._copied_config()
        config.update(max_new_tokens=1024, do_sample=False)

        sequence: torch.Tensor = self.__generate_sequence_greedy(inputs, config)[:, inputs["input_ids"].size(1):]
        return LlmAssistantMessage(self._processor.batch_decode(sequence, skip_special_tokens=True)[0], None)

    @torch.no_grad
    def chat(self, messages: Sequence[LlmMessage], max_new_tokens: int) -> LlmAssistantMessage:
        inputs: object = self._processor.apply_chat_template(
            self._messages_to_transformers(messages),
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        assert isinstance(inputs, transformers.feature_extraction_utils.BatchFeature)
        inputs = inputs.to(self._device, dtype=torch.bfloat16)
        
        messages_modified: Iterable[LlmMessage] = self.__sampler.apply_augmentation(self.__chat_greedy, messages, None)
        inputs_modified: object = self._processor.apply_chat_template(
            self._messages_to_transformers(messages_modified),
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        assert isinstance(inputs_modified, transformers.feature_extraction_utils.BatchFeature)
        inputs_modified = inputs_modified.to(self._device, dtype=torch.bfloat16)

        config: GenerationConfig = self._copied_config()
        config.update(max_new_tokens=max_new_tokens)

        mm_token_type_ids: Tensor2d = Tensor2d(torch.tensor(
            self._processor.create_mm_token_type_ids(inputs["input_ids"].tolist()),
            device=self._device
        ))

        sequence: torch.Tensor = self.__generate_sequence_selfaug(
            inputs, inputs_modified, mm_token_type_ids, config)[:, inputs["input_ids"].size(1):]
        return LlmAssistantMessage(self._processor.batch_decode(sequence, skip_special_tokens=True)[0], None)
    
    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        raise NotImplementedError("Self-Aug does not work on one single input. "
                                  "Please use the overridden 'chat' method for Self-Aug, "
                                  "which will call '__generate_sequence_greedy' and '__generate_sequence_selfaug' internally.")
    
    def __generate_sequence_greedy(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        return getattr(self._model, "generate")(
            **inputs,
            generation_config=generation_config
        )

    def __generate_sequence_selfaug(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        inputs_modified: transformers.feature_extraction_utils.BatchFeature, 
        mm_token_type_ids: Tensor2d,
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        parameter: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs, generation_config)
        parameter_modified: InternVl3d5Base._Parameter = self._sample_preparer_prepare_for(inputs_modified, generation_config)
        
        model_kwargs: dict = {**parameter}
        model_kwargs_selfaug: dict = {**parameter_modified}
        shared: dict[str, Any] = {}
        key: str
        for key in ["input_ids", "logits_processor", "stopping_criteria", "generation_config", "synced_gpus"]:
            shared[key] = model_kwargs.pop(key)
            model_kwargs_selfaug.pop(key)

        return self.__sampler.sample(
            self._model, 
            **shared, 
            streamer=None, 
            random=None, 
            model_kwargs=model_kwargs, 
            model_kwargs_selfaug=model_kwargs_selfaug,
            gate=self.__gate.new(mm_token_type_ids)
        )
