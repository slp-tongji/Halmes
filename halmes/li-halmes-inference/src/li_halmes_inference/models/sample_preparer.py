import inspect
from typing import Any, Callable, Protocol
import warnings

from li_basic_utilities import StaticClass
import torch
from torch.types import Number
import transformers
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel


class SamplePreparer(StaticClass):
    __modes: dict[GenerationMode, str] = {
        GenerationMode.SAMPLE: "_sample",
        GenerationMode.GREEDY_SEARCH: "_sample",
        GenerationMode.BEAM_SEARCH: "_beam_search",
        GenerationMode.BEAM_SAMPLE: "_beam_search",
        GenerationMode.ASSISTED_GENERATION: "_assisted_decoding",
    }

    @staticmethod
    def prepare_for(
        model: TransformersGenerativePreTrainedModel,
        generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        # https://github.com/huggingface/transformers/blob/1897dd055ddcaddd80ed83bc45bbf91181de4804/src/transformers/generation/utils.py

        assert "inputs" not in kwargs
        assert "logits_processor" not in kwargs
        assert "stopping_criteria" not in kwargs
        assert "prefix_allowed_tokens_fn" not in kwargs
        assert "assistant_model" not in kwargs
        assert "streamer" not in kwargs
        assert "negative_prompt_ids" not in kwargs
        assert "negative_prompt_attention_mask" not in kwargs
        assert "custom_generate" not in kwargs
        assert "synced_gpus" not in kwargs
        assert "cache_implementation" not in kwargs

        trust_remote_code: bool | None = kwargs.pop("trust_remote_code", None)
        generation_mode_kwargs: dict[str, Any] = model._extract_generation_mode_kwargs(
            custom_generate=None,
            kwargs=kwargs,
            synced_gpus=None,
            assistant_model=None,
            streamer=None
        )

        has_default_max_length: bool = (
            kwargs.get("max_length") is None
            and (generation_config is None or getattr(generation_config, "max_length") is None)
            and model.generation_config.max_length is None
        )
        has_default_min_length: bool = (
            kwargs.get("min_length") is None
            and (generation_config is None or getattr(generation_config, "min_length") is None)
            and model.generation_config.min_length is None
        )
        model_kwargs: dict
        generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
        assert generation_config is not None

        generation_mode: GenerationMode = generation_config.get_generation_mode(assistant_model=None)
        deprecated_mode_repo: str | None = model._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate=None)
        assert deprecated_mode_repo is None

        decoding_method: Callable = getattr(type(model), SamplePreparer.__modes[generation_mode])

        model._validate_model_kwargs(model_kwargs.copy())
        model._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        logits_processor: LogitsProcessorList = LogitsProcessorList()
        stopping_criteria: StoppingCriteriaList = StoppingCriteriaList()

        accepts_attention_mask: bool = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
        kwargs_has_attention_mask: bool = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor: torch.Tensor
        model_input_name: str | None
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs=None, 
            bos_token_id=getattr(generation_config, "bos_token_id"), 
            model_kwargs=model_kwargs
        )
        if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
            generation_mode_kwargs["inputs_tensor"] = inputs_tensor
        batch_size: int = inputs_tensor.shape[0]

        device: torch.device = inputs_tensor.device
        model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not model.config.is_encoder_decoder:
            if generation_config._pad_token_tensor is not None and batch_size > 1 and len(inputs_tensor.shape) == 2:
                attention_mask: torch.Tensor | None = model_kwargs.get("attention_mask", None)
                if attention_mask is not None and attention_mask.shape == inputs_tensor.shape:
                    has_right_padding: Number | torch.Tensor = torch.any(attention_mask[:, -1] == 0).item()
                else:
                    has_right_padding = torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
                if has_right_padding:
                    raise RuntimeError(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

        if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            setattr(generation_config, "use_cache", True)

        if not kwargs_has_attention_mask and not model.config.is_encoder_decoder and accepts_attention_mask:
            model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        kwargs_has_position_ids: bool = model_kwargs.get("position_ids", None) is not None
        accepts_position_ids: bool = "position_ids" in set(inspect.signature(model.forward).parameters.keys())
        if not kwargs_has_position_ids and accepts_position_ids and not model.config.is_encoder_decoder:
            model_kwargs["position_ids"] = model._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if model.config.is_encoder_decoder:
            input_ids: torch.Tensor
            input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(getattr(generation_config, "num_beams"), getattr(generation_config, "num_return_sequences")),
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )

        if getattr(generation_config, "token_healing"):
            input_ids = model.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

        input_ids_length: int = input_ids.shape[1]
        generation_config = model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if model._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        max_cache_length: int = getattr(generation_config, "max_length") - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not model.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        model._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        if model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        prepared_logits_processor: LogitsProcessorList = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )
        prepared_stopping_criteria: StoppingCriteriaList = model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        model_kwargs["use_cache"] = getattr(generation_config, "use_cache")

        return {
            "input_ids": input_ids,
            "logits_processor": prepared_logits_processor,
            "stopping_criteria": prepared_stopping_criteria,
            "generation_config": generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        }