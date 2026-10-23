from pathlib import Path
from typing import Iterable, Sequence

from li_halmes_configurations import HalmesTargetLlm
from li_image_operating import InMemoryImage
from li_llm_messages import LlmAssistantMessage, LlmMessage, LlmSystemMessage
from statictorch import Tensor1d, Tensor3d
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
import transformers
from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration
from transformers.models.internvl.processing_internvl import InternVLProcessor

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput

from li_halmes_generate_data.supported_llm import SupportedLlm


class Qwen3Vl(SupportedLlm):
    def __init__(self, target: HalmesTargetLlm, device: torch.device) -> None:
        assert target["model_type"] == "qwen3_vl"

        super().__init__(target["id"])

        self.__device = device
        model: object = Qwen3VLForConditionalGeneration.from_pretrained(
            target["model_path"],
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="eager"
        )
        assert isinstance(model, Qwen3VLForConditionalGeneration)
        self.__model: InternVLForConditionalGeneration = getattr(model, "to")(self.__device).eval()

        self.__processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(target["model_path"])

        self.__config = GenerationConfig(
            do_sample=target["do_sample"],
            temperature=target["temperature"],
            top_p=target["top_p"],
            top_k=target["top_k"],

            return_dict_in_generate=True,
            output_attentions=True
        )

    def __messages_to_transformers(self, messages: Iterable[LlmMessage]):
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

    def chat(self, messages: Sequence[LlmMessage], max_new_tokens: int) -> SupportedLlm.Output:
        inputs: object = self.__processor.apply_chat_template(
            self.__messages_to_transformers(messages),
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        assert isinstance(inputs, transformers.feature_extraction_utils.BatchFeature)
        inputs = inputs.to(self.__device, dtype=torch.bfloat16)
        
        mm_token_type_ids: torch.Tensor = inputs["mm_token_type_ids"]

        config: GenerationConfig = self._copy_config(self.__config)
        config.update(max_new_tokens=max_new_tokens)

        outputs: object = getattr(self.__model, "generate")(
            **inputs,
            generation_config=config
        )
        assert isinstance(outputs, GenerateDecoderOnlyOutput)

        assert outputs.attentions is not None
        return {
            "mm_token_type_ids": Tensor1d(mm_token_type_ids.squeeze(0)),
            "sequence": Tensor1d(outputs.sequences.squeeze(0)),
            "attentions": [[Tensor3d(a.squeeze(0)) for a in l] for l in outputs.attentions]
        }