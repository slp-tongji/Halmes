import abc
from typing import Iterable, Sequence, TypedDict
import typing

from li_halmes_configurations import HalmesTargetLlm
from li_image_operating import InMemoryImage
from li_llm_messages import LlmAssistantMessage, LlmMessage, LlmSystemMessage
import torch
from transformers import AutoProcessor, DynamicCache, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
import transformers

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from li_halmes_inference.models.sample_preparer import SamplePreparer
from li_halmes_inference.supported_llm import SupportedLlm


class Qwen3VlBase(SupportedLlm, abc.ABC):
    @abc.abstractmethod
    def _generate_sequence(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        ...

    def __init__(self, variant_id: str,
                 target: HalmesTargetLlm, device: torch.device, 
                 replace_model: Qwen3VLForConditionalGeneration | None,
                 replace_config: GenerationConfig | None) -> None:
        assert target["model_type"] == "qwen3_vl"

        super().__init__(variant_id)

        self.__device = device
        if replace_model is None:
            replace_model = Qwen3VLForConditionalGeneration.from_pretrained(
                target["model_path"],
                dtype=torch.bfloat16,
                device_map=None,
                attn_implementation="eager"
            )
        self.__model: Qwen3VLForConditionalGeneration = getattr(replace_model, "to")(self.__device).eval()

        self.__processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(target["model_path"])

        if replace_config is None:
            replace_config = GenerationConfig(
                do_sample=target["do_sample"],
                temperature=target["temperature"],
                top_p=target["top_p"],
                top_k=target["top_k"]
            )
        self.__config = replace_config

    @property
    def _model(self):
        return self.__model

    @property
    def _processor(self):
        return self.__processor

    @property
    def _device(self):
        return self.__device

    def _copied_config(self):
        return self._copy_config(self.__config)

    def _messages_to_transformers(self, messages: Iterable[LlmMessage]):
        result: list = []
        message: LlmMessage
        for message in messages:
            if isinstance(message, LlmSystemMessage):
                result.append({
                    "role": "system",
                    "content": message.content
                })
            if isinstance(message, LlmAssistantMessage):
                result.append({
                    "role": "assistant",
                    "content": message.content
                })
            content: list = []
            block: str | InMemoryImage
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                    continue
                content.append({
                    "type": "image", 
                    "image": block.data_url
                })
            result.append({"role": "user", "content": content})
        return result

    class _Parameter(TypedDict):
        input_ids: torch.LongTensor
        logits_processor: LogitsProcessorList
        stopping_criteria: StoppingCriteriaList
        generation_config: GenerationConfig
        synced_gpus: bool
        attention_mask: torch.Tensor
        mm_token_type_ids: torch.Tensor
        position_ids: torch.Tensor
        logits_to_keep: int
        past_key_values: DynamicCache
        use_cache: bool
        pixel_values: torch.Tensor
        image_grid_thw: torch.Tensor

    def _sample_preparer_prepare_for(
        self, 
        inputs: transformers.feature_extraction_utils.BatchFeature, 
        generation_config: GenerationConfig
    ):
        return typing.cast(
            Qwen3VlBase._Parameter, 
            SamplePreparer.prepare_for(self.__model, **inputs, generation_config=generation_config)
        )

    @torch.no_grad
    def chat(self, messages: Sequence[LlmMessage], max_new_tokens: int) -> LlmAssistantMessage:
        inputs: object = self.__processor.apply_chat_template(
            self._messages_to_transformers(messages),
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        assert isinstance(inputs, transformers.feature_extraction_utils.BatchFeature)
        inputs = inputs.to(self.__device, dtype=torch.bfloat16)
        
        config: GenerationConfig = self._copy_config(self.__config)
        config.update(max_new_tokens=max_new_tokens)

        sequence: torch.Tensor = self._generate_sequence(inputs, config)[:, inputs["input_ids"].size(1):]
        return LlmAssistantMessage(self.__processor.batch_decode(sequence, skip_special_tokens=True)[0], None)
    