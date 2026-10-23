import math
from typing import Any, Callable, Literal
import typing

from torch import LongTensor, Tensor

import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.samplers.vcd_sampler import VcdSampler
from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel
from li_halmes_inference.supported_gate import SupportedGate


class M3idSampler:
    def __init__(self, 
                 lambda_: float = 0.02, 
                 alpha: float = 0.3, 
                 source: Literal["base", "zeros"] | VcdSampler = "zeros") -> None:
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__source: Literal["base", "zeros"] | VcdSampler = source
    
    def modify_image(self, pixel_values: Tensor, random: torch.Generator | None):
        if self.__source == "zeros":
            return torch.zeros_like(pixel_values)
        elif self.__source == "base":
            return pixel_values.clone()
        else:
            return self.__source.add_diffusion_noise(pixel_values, random)

    def __process_logits(self, t: int, next_token_logits: Tensor, next_token_logits_m3id: Tensor):
        gamma: float = math.exp(-self.__lambda * t)
        lc: Tensor = torch.log_softmax(next_token_logits, dim=-1)
        lu: Tensor = torch.log_softmax(next_token_logits_m3id, dim=-1)
        indicator: Tensor = (lc.max(dim=-1, keepdim=True).values < math.log(self.__alpha)).to(lc.dtype)
        return lc + indicator * ((1 - gamma) / gamma) * (lc - lu)
    
    def sample(
        self,
        model: TransformersGenerativePreTrainedModel,
        input_ids: LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: transformers.generation.streamers.BaseStreamer | None,
        random: torch.Generator | None,
        model_kwargs: dict[str, Any],
        model_kwargs_m3id: dict[str, Any],
        gate: SupportedGate
    ) -> LongTensor:
        assert not getattr(generation_config, "output_attentions")
        assert not getattr(generation_config, "output_hidden_states")
        assert not getattr(generation_config, "output_scores")
        assert not getattr(generation_config, "output_logits")
        assert not getattr(generation_config, "return_dict_in_generate")

        model_kwargs["output_attentions"] = True

        pad_token_id: Tensor | None = generation_config._pad_token_tensor

        has_eos_stopping_criteria: bool = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample: bool | None = getattr(generation_config, "do_sample")

        batch_size: int = input_ids.shape[0]
        this_peer_finished: bool | Tensor = False
        unfinished_sequences: Tensor = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        model_forward: Callable = (
            model.get_compiled_call(getattr(generation_config, "compile_config"))
            if model._valid_auto_compile_criteria(model_kwargs, generation_config)
            else model.__call__
        )

        prefill_consumed: bool = False
        outputs: Any = model._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )
        outputs_m3id: Any = model._prefill(
            input_ids,
            generation_config,
            model_kwargs_m3id,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )
        
        # This version follows
        # https://github.com/LijunZhang01/Octopus/blob/master/avisc_utils/avisc_sample.py
        t_m3id: int = 1

        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length: int | None = 1 if model_kwargs["use_cache"] else None

                model_inputs: dict[str, Any] = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                with model._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)

                model_inputs_m3id: dict[str, Any] = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs_m3id
                )
                with model._optimize_model_for_decode():
                    outputs_m3id = model_forward(**model_inputs_m3id, return_dict=True)
            prefill_consumed = True

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            model_kwargs_m3id = model._update_model_kwargs_for_generation(
                outputs_m3id,
                model_kwargs_m3id,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            t_m3id += 1

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits: Tensor = outputs.logits[:, -1, :].to(copy=True, 
                                                                    dtype=torch.float32, device=input_ids.device)
            next_token_logits_m3id: Tensor = outputs_m3id.logits[:, -1, :].to(copy=True, 
                                                                              dtype=torch.float32, device=input_ids.device)
            
            if gate.next(outputs.attentions):
                next_token_logits = self.__process_logits(t_m3id, next_token_logits, next_token_logits_m3id)

            next_token_scores: Tensor = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs: Tensor = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens: Tensor = torch.multinomial(probs, num_samples=1, generator=random).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                assert pad_token_id is not None
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = typing.cast(LongTensor, 
                                    torch.cat([input_ids, next_tokens[:, None]], dim=-1))
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Here is an type error but it is the original logic in GenerationMixin.
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores=None) # pyright: ignore[reportArgumentType]
            this_peer_finished = unfinished_sequences.max() == 0

            outputs = None
            outputs_m3id = None

        if streamer is not None:
            streamer.end()

        return input_ids
    