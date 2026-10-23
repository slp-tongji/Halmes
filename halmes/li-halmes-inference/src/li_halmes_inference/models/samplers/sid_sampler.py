from typing import Any, Callable
import typing

from statictorch import Tensor2d
from torch import LongTensor, Tensor

import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel
from li_halmes_inference.supported_gate import SupportedGate


class SidSampler:
    def __init__(self, 
                 keep_ratio: float = 0.1,  # save the lowest 10% attention scores (most relevant tokens)
                 agg_layer: int = 2,       # extract attention from the 2nd layer (0-based index)
                 alpha: float = 1.0, 
                 beta: float | None = 0.1) -> None:
        self.__keep_ratio = keep_ratio
        self.__agg_layer = agg_layer
        self.__alpha = alpha
        self.__beta = beta

    def __process_logits(self, next_token_logits: Tensor, next_token_logits_sid: Tensor):
        result: Tensor = (1 + self.__alpha) * next_token_logits - self.__alpha * next_token_logits_sid
        if self.__beta is not None:
            cutoff: Tensor = torch.tensor(self.__beta, 
                                          device=next_token_logits.device,
                                          dtype=next_token_logits.dtype)
            cutoff = torch.log(cutoff) + next_token_logits.max(dim=-1, keepdim=True).values
            result = result.masked_fill(next_token_logits < cutoff, -float("inf"))
        return result
    
    def __apply_sid_mask(self, attentions: tuple[Tensor, ...], original_mask: Tensor, image_token_indices: list[Tensor]) -> Tensor:
        # 1. Extract Attention: [batch_size, num_heads, q_len, kv_len]
        target_attn: Tensor = attentions[self.__agg_layer] 
        
        # 2. Average over heads and query positions to get a single attention score per kv token
        # [batch_size, kv_len]
        avg_attn: Tensor = target_attn.mean(dim=1)[:, -1, :] 
        
        modified_mask: Tensor = original_mask.clone()
        
        zero_val: bool | int = False if original_mask.dtype == torch.bool else 0
        one_val: bool | int = True if original_mask.dtype == torch.bool else 1

        batch_size: int = avg_attn.size(0)
        b: int
        for b in range(batch_size):
            b_image_indices: Tensor = image_token_indices[b]
            if len(b_image_indices) == 0:
                continue
                
            # 3. Extract the attention scores for the current generated token with respect to image tokens
            img_attn_scores: Tensor = avg_attn[b, b_image_indices]
            
            # 4. Calculate the number of tokens to keep
            k: int = max(1, int(len(img_attn_scores) * self.__keep_ratio))
            
            # 5. Find the indices of the top-k lowest attention scores (most relevant tokens)
            lowest_attn_local_indices: Tensor = img_attn_scores.topk(k, largest=False).indices
            keep_absolute_indices: Tensor = b_image_indices[lowest_attn_local_indices]
            
            # 6. Modify the attention mask
            if original_mask.dim() == 2:
                modified_mask[b, b_image_indices] = zero_val
                modified_mask[b, keep_absolute_indices] = one_val
            elif original_mask.dim() == 4:
                modified_mask[b, :, :, b_image_indices] = zero_val
                modified_mask[b, :, :, keep_absolute_indices] = one_val

        return modified_mask

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
        model_kwargs_sid: dict[str, Any],
        gate: SupportedGate,
        mm_token_type_ids: Tensor2d
    ) -> LongTensor:
        assert not getattr(generation_config, "output_attentions")
        assert not getattr(generation_config, "output_hidden_states")
        assert not getattr(generation_config, "output_scores")
        assert not getattr(generation_config, "output_logits")
        assert not getattr(generation_config, "return_dict_in_generate")

        model_kwargs["output_attentions"] = True

        image_token_indices: list[Tensor] = []
        b: int
        for b in range(input_ids.shape[0]):
            indices: Tensor = (mm_token_type_ids[b] == 1).nonzero(as_tuple=True)[0]
            image_token_indices.append(indices)

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
    
        outputs_sid: Any = model._prefill(
            input_ids,
            generation_config,
            model_kwargs_sid,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )

        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length: int | None = 1 if model_kwargs["use_cache"] else None

                model_inputs: dict[str, Any] = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                with model._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)

                model_inputs_sid: dict[str, Any] = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs_sid
                )
                
                # ====== SID Core ======
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    model_inputs_sid["attention_mask"] = self.__apply_sid_mask(
                        outputs.attentions, 
                        model_inputs_sid["attention_mask"], 
                        image_token_indices
                    )
                # ========================================

                with model._optimize_model_for_decode():
                    outputs_sid = model_forward(**model_inputs_sid, return_dict=True)
            
            prefill_consumed = True

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            model_kwargs_sid = model._update_model_kwargs_for_generation(
                outputs_sid,
                model_kwargs_sid,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits: Tensor = outputs.logits[:, -1, :].to(copy=True, 
                                                                    dtype=torch.float32, device=input_ids.device)
            next_token_logits_sid: Tensor = outputs_sid.logits[:, -1, :].to(copy=True, 
                                                                            dtype=torch.float32, device=input_ids.device)
            
            if gate.next(outputs.attentions):
                next_token_logits = self.__process_logits(next_token_logits, next_token_logits_sid)

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

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores=None) # pyright: ignore[reportArgumentType]
            this_peer_finished = unfinished_sequences.max() == 0

            outputs = None
            outputs_sid = None

        if streamer is not None:
            streamer.end()

        return input_ids