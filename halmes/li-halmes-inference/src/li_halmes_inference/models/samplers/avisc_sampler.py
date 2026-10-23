from functools import cache
from typing import Any, Callable, Literal, Sequence, TypedDict
import typing
from uuid import uuid4

from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d, Tensor4d
from torch import LongTensor, Tensor

import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.samplers.vcd_sampler import VcdSampler
from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel
from li_halmes_inference.supported_gate import SupportedGate


class AviscSampler:
    def __init__(self, 
                 layer_gamma: float = 0.5, 
                 lambda_: float = 1.,
                 alpha: float = 1.0, beta: float | None = 0.1,
                 source: Literal["zeros", "ones"] | VcdSampler = "zeros") -> None:
        self.__lambda = lambda_
        self.__layer_gamma = layer_gamma
        self.__alpha = alpha
        self.__beta = beta
        self.__source: Literal["zeros", "ones"] | VcdSampler = source

    def __top_p(self, values: Tensor1d, top_p: float):
        sorted: torch.return_types.sort = torch.sort(values, descending=True)
        values = Tensor1d(sorted.values)
        indices: Tensor1d = Tensor1d(sorted.indices)

        values = Tensor1d(torch.cumsum(values, dim=0))
        k: int = int((values < top_p).sum()) + 1
        
        return Tensor1d(values[:k]), Tensor1d(indices[:k])

    def __get_mask(self, 
                   attentions_original: Sequence[Tensor4d], 
                   image_mask: Tensor2d,
                   random: torch.Generator | None) -> 'AviscSampler.Mask':
        result: Tensor2d = Tensor2d(torch.zeros_like(image_mask, 
                                                     dtype=torch.bool, 
                                                     device=attentions_original[0].device))

        batch: int
        for batch in range(len(image_mask)):
            image_attentions: list[Tensor1d] = []

            attention: torch.Tensor
            for attention in attentions_original:
                image_attentions.append(Tensor1d(attention.mean(dim=1)[batch, -1, image_mask[batch, :]]))

            attention_portion: Tensor1d = Tensor1d(torch.stack([a.sum() for a in image_attentions], dim=0))
            attention_portion = Tensor1d(attention_portion / attention_portion.sum().clamp_min(1e-12))

            top_p: Tensor1d
            _, top_p = self.__top_p(attention_portion, self.__layer_gamma)
            image_attention_tensor_by_layer: Tensor2d = Tensor2d(torch.stack([
                image_attentions[i]
                for i in top_p
            ], dim=0))
            
            image_attention_tensor: Tensor1d = Tensor1d(image_attention_tensor_by_layer.mean(dim=0))

            threshold: Tensor0d = Tensor0d(image_attention_tensor.mean() + 
                                           image_attention_tensor.std() * self.__lambda)
            selected_mask: Tensor1d = Tensor1d(image_attention_tensor < threshold)
            
            result[batch, image_mask[batch, :]] = selected_mask
        
        if self.__source == "zeros":
            return AviscSampler.Mask(result, 0)
        elif self.__source == "ones":
            return AviscSampler.Mask(result, 1)
        else:
            return AviscSampler.Mask(result, random)

    def __process_logits(self, next_token_logits: Tensor, next_token_logits_avisc: Tensor):
        if torch.isnan(next_token_logits_avisc).any():
            next_token_logits_avisc = next_token_logits
        result: Tensor = (1 + self.__alpha) * next_token_logits - self.__alpha * next_token_logits_avisc
        if self.__beta is not None:
            cutoff: Tensor = torch.tensor(self.__beta, 
                                          device=next_token_logits.device,
                                          dtype=next_token_logits.dtype)
            cutoff = torch.log(cutoff) + next_token_logits.max(dim=-1, keepdim=True).values
            result = result.masked_fill(next_token_logits < cutoff, -float("inf"))
        return result
    
    class Mask:
        def __init__(self, mask: Tensor2d, random: torch.Generator | None | Literal[0, 1]) -> None:
            self.__mask = mask
            self.__random = random

        def __fill(self, tensor: Tensor, mask: Tensor):
            if isinstance(self.__random, int):
                return tensor.masked_fill(mask, self.__random)
            else:
                return torch.where(mask, 
                                   torch.randn_like(tensor, generator=self.__random), 
                                   tensor)

        def mask_inputs(self, inputs_embeds: Tensor3d):
            assert self.__mask.size(0) == inputs_embeds.size(0)
            assert self.__mask.size(1) == inputs_embeds.size(1)

            mask: Tensor3d = Tensor3d(self.__mask.unsqueeze(-1))
            return Tensor3d(self.__fill(inputs_embeds, mask))

        def mask_deepstack(self, deepstack_embeds: list[Tensor2d], mask: Tensor2d):
            mask = Tensor2d(self.__mask[mask].unsqueeze(-1))
            result: list[Tensor2d] = []
            layer: Tensor2d
            for layer in deepstack_embeds:
                result.append(Tensor2d(self.__fill(layer, mask)))
            return result

    class Parameter(TypedDict):
        mask: 'AviscSampler.Mask'

    def sample(
        self,
        model_that_masks_image: TransformersGenerativePreTrainedModel,
        input_ids: LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: transformers.generation.streamers.BaseStreamer | None,
        random: torch.Generator | None,
        model_kwargs: dict[str, Any],
        model_kwargs_avisc: dict[str, Any],
        gate: SupportedGate,
        mm_token_type_ids: Tensor2d
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
            model_that_masks_image.get_compiled_call(getattr(generation_config, "compile_config"))
            if model_that_masks_image._valid_auto_compile_criteria(model_kwargs, generation_config)
            else model_that_masks_image.__call__
        )

        prefill_consumed: bool = False

        outputs: Any = model_that_masks_image._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )

        model_kwargs_avisc["avisc"] = {
            "mask": self.__get_mask(outputs.attentions, Tensor2d(mm_token_type_ids == 1), None)
        }
        outputs_avisc: Any = model_that_masks_image._prefill(
            input_ids,
            generation_config,
            model_kwargs_avisc,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )
        del model_kwargs_avisc["avisc"]

        while model_that_masks_image._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length: int | None = 1 if model_kwargs["use_cache"] else None

                model_inputs: dict[str, Any] = model_that_masks_image.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                with model_that_masks_image._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)

                model_inputs_avisc: dict[str, Any] = model_that_masks_image.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs_avisc
                )
                with model_that_masks_image._optimize_model_for_decode():
                    outputs_avisc = model_forward(**model_inputs_avisc, return_dict=True)
            prefill_consumed = True

            model_kwargs = model_that_masks_image._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model_that_masks_image.config.is_encoder_decoder,
            )
            model_kwargs_avisc = model_that_masks_image._update_model_kwargs_for_generation(
                outputs_avisc,
                model_kwargs_avisc,
                is_encoder_decoder=model_that_masks_image.config.is_encoder_decoder,
            )

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits: Tensor = outputs.logits[:, -1, :].to(copy=True, 
                                                                    dtype=torch.float32, device=input_ids.device)
            next_token_logits_avisc: Tensor = outputs_avisc.logits[:, -1, :].to(copy=True, 
                                                                            dtype=torch.float32, device=input_ids.device)
            if gate.next(outputs.attentions):
                next_token_logits = self.__process_logits(next_token_logits, next_token_logits_avisc)

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
            outputs_avisc = None

        if streamer is not None:
            streamer.end()

        return input_ids
    