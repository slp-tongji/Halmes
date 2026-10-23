from typing import Any, Callable, Literal
import typing

from li_basic_utilities import StaticClass
from torch import LongTensor, Tensor

import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel


class BeamSearchSampler(StaticClass):
    """
    This is the original `GenerationMixin._beam_search` from `transformers`, but with some simplification.
    Normally, one should use `transformers` directly; this type is mainly for reference when implementing other samplers.
    """
    @staticmethod
    def sample(
        model: TransformersGenerativePreTrainedModel,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        vocab_size: int,
        **model_kwargs: Any
    ) -> Tensor:
        assert not getattr(generation_config, "output_attentions")
        assert not getattr(generation_config, "output_hidden_states")
        assert not getattr(generation_config, "output_scores")
        assert not getattr(generation_config, "output_logits")
        assert not getattr(generation_config, "return_dict_in_generate")
        
        pad_token_id: Tensor | None = generation_config._pad_token_tensor
        eos_token_id: Tensor | None = generation_config._eos_token_tensor
        do_sample: bool | None = getattr(generation_config, "do_sample")
        early_stopping: bool | str | None = getattr(generation_config, "early_stopping")
        length_penalty: float | None = getattr(generation_config, "length_penalty")
        max_length: int | None = getattr(generation_config, "max_length")
        num_beams: int | None = getattr(generation_config, "num_beams")
        num_return_sequences: int | None = getattr(generation_config, "num_return_sequences")

        assert max_length is not None
        assert num_beams is not None

        batch_size_unflattened: int
        cur_len: int
        batch_size_unflattened, cur_len = input_ids.shape[:2]

        batch_size: int
        batch_size = batch_size_unflattened // num_beams

        decoder_prompt_len: int = cur_len
        this_peer_finished: bool = False

        n_eos_tokens: int = eos_token_id.shape[0] if eos_token_id is not None else 0
        beams_to_keep: int = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask: Tensor = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0,
        ).to(input_ids.device)

        assert not getattr(generation_config, "low_memory")

        output_fill_value: int = int(pad_token_id or eos_token_id[0]) if eos_token_id is not None else -1
        running_sequences: Tensor = torch.full(
            size=(batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=torch.int64,
            device=input_ids.device,
        )
        running_sequences[:, :, :cur_len] = model._unflatten_beam_dim(input_ids, batch_size, num_beams)
        sequences: Tensor = running_sequences.detach().clone()

        running_beam_scores: Tensor = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores: Tensor = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

        is_sent_finished: Tensor = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)
        is_early_stop_heuristic_unsatisfied: Tensor = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)
        next_token_hits_stopping_criteria: Tensor = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
        )

        running_beam_indices: Tensor = torch.full(
            (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
        )
        beam_indices: Tensor = running_beam_indices.detach().clone()

        flat_running_sequences: Tensor = input_ids
        prefill_consumed: bool = False
        model_outputs: Any = model._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not getattr(generation_config, "is_assistant"),
        )

        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                flat_running_sequences = model._flatten_beam_dim(running_sequences[:, :, :cur_len])
                next_sequence_length: Literal[1] | None = 1 if model_kwargs["use_cache"] else None
                model_inputs: Any = model.prepare_inputs_for_generation(
                    flat_running_sequences, next_sequence_length=next_sequence_length, **model_kwargs
                )
                model_outputs = model(**model_inputs, return_dict=True)
            prefill_consumed = True

            model_kwargs = model._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            logits: Tensor = model_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            log_probs: Tensor = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = logits_processor(flat_running_sequences, log_probs)

            model_outputs = None

            log_probs = model._unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

            topk_log_probs: Tensor
            topk_running_sequences: Tensor
            topk_running_beam_indices: Tensor
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = model._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # Here is an type error but it is the original logic in GenerationMixin.
            next_token_hits_stopping_criteria = stopping_criteria(
                input_ids=model._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),
                scores=None,    # pyright: ignore[reportArgumentType]
            )
            next_token_hits_stopping_criteria = model._unflatten_beam_dim(
                next_token_hits_stopping_criteria, batch_size, beams_to_keep
            )

            running_sequences, running_beam_scores, running_beam_indices = model._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            sequences, beam_scores, beam_indices, is_sent_finished = model._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

            if model_kwargs.get("past_key_values") is not None:
                beam_idx: Tensor = model._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
                if hasattr(model, "_reorder_cache"):
                    model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)
                else:
                    model_kwargs["past_key_values"].reorder_cache(beam_idx)

            cur_len = cur_len + 1
            is_early_stop_heuristic_unsatisfied = model._check_early_stop_heuristic(
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                running_beam_scores=running_beam_scores,
                beam_scores=beam_scores,
                is_sent_finished=is_sent_finished,
                cur_len=cur_len,
                max_length=max_length,
                decoder_prompt_len=decoder_prompt_len,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )
            this_peer_finished = not model._beam_search_has_unfinished_sequences(
                is_early_stop_heuristic_unsatisfied,
                is_sent_finished,
                next_token_hits_stopping_criteria,
                early_stopping,
            )

        sequences = model._flatten_beam_dim(sequences[:, :num_return_sequences, :])
        beam_scores = model._flatten_beam_dim(beam_scores[:, :num_return_sequences])
        beam_indices = model._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        max_generated_length: Tensor = ((beam_indices + 1).bool()).sum(dim=1).max()
        output_length: Tensor = decoder_prompt_len + max_generated_length
        sequences = sequences[:, :output_length]
        beam_indices = beam_indices[:, :max_generated_length]

        return sequences