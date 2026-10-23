import math
from typing import Any, Callable, Iterable, Sequence
import typing
import warnings

import PIL
from PIL import ImageDraw
import PIL.Image
from li_image_operating import InMemoryImage
from li_llm_messages import LlmAssistantMessage, LlmMessage, LlmMessages, LlmUserMessage
from statictorch import Tensor1d, Tensor2d
from torch import LongTensor, Tensor

import torch
import torchvision
import torchvision.transforms.functional
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from li_halmes_inference.models.samplers.vcd_sampler import VcdSampler
from li_halmes_inference.models.transformers_generative_pre_trained_model import TransformersGenerativePreTrainedModel
from li_halmes_inference.supported_gate import SupportedGate


class SelfaugSampler:
    def __init__(self, 
                 crop_ratio: float = 2., 
                 mask_ratio: float = 2., 
                 noise: VcdSampler = VcdSampler(), 
                 alpha: float = 1.,
                 beta: float = 0.1,
                 tau: float | None = 0.5) -> None:
        self.__crop_ratio: float = crop_ratio
        self.__mask_ratio: float = mask_ratio
        self.__noise: VcdSampler = noise
        self.__alpha: float = alpha
        self.__beta: float = beta
        self.__tau: float | None = tau

    def __request_augmentation(self,
                               chat_greedy: Callable[[Sequence[LlmMessage]], LlmAssistantMessage], 
                               query: str):
        prompt: str = f"""You are an expert data augmentation analyst. Your task is to select the single most semantically disruptive image augmentation that most effectively invalidates the question's premise or prevents a confident answer. Provide a clear reason explaining why the augmentation is chosen, then state your final choice.

## Augmentations and Their Effects ##
- Vertical flip: Flips image top-to-bottom. Disrupts questions about “above”, “below”, “under” or reading orientation.
- Color inversion: Replaces each color with its complement. Disrupts questions relying on accurate color identification.
- Random crop: Removes random parts of the image. Disrupts questions requiring global context or peripheral objects.
- Random mask: Occludes portions of the image. Disrupts object presence, count, or attribute recognition.
- Noise: Adds visual distortion. Disrupts questions requiring small details, texture, or text clarity.
- Horizontal flip: Flips the image left-to-right. Disrupts questions about left/right positioning and left-to-right text reading.

## Examples ##
Question: "Is the mirror above the TV?"
Reason: The question focuses on vertical positioning. Vertical flip reverses top and bottom, making “above” mean “below,” invalidating the question. Other augmentations don't affect vertical relationships.
Choice: vertical flip

Question: "Is this photo taken indoors?"
Reason: The question requires identifying a specific environmental context. Random crop may exclude key background elements like trees, invalidating the question. Flips, color inversion, noise, and random mask don't directly affect scene context.
Choice: random crop

Question: "Are there any green beans in the image?"
Reason: The question requires identifying a specific color. Color inversion changes green to its complement, invalidating the question. Flips, noise, random mask, and random crop don't target color directly.
Choice: color inversion

Question: "How many people are in the image?"
Reason: The question requires counting visible people. Random mask can completely obscure one or more people, making the exact count impossible. Noise obscures details but typically doesn't hide entire objects, allowing approximate counting. Flips and color inversion don't affect object visibility or count.
Choice: random mask

Question: "Is the cat on the right side of the laptop?"
Reason: The question relies on horizontal positioning. Horizontal flip reverses left and right, making “right” mean “left”, invalidating the question. Other augmentations don't target horizontal positions.
Choice: horizontal flip

Question: "Does this artwork exist in the form of painting?"
Reason: The question requires identifying the texture of the artwork. Noise obscures fine details, making it hard to identify the medium. Other augmentations don't target texture details.
Choice: noise

## Your Answer ##
If multiple augmentations could disrupt the question, select the one whose effect is most direct and unambiguous. You must choose one of the given augmentations following the "Reason:" and "Choice:" format.

Question: "{query}"
"""
        response: LlmAssistantMessage = chat_greedy([LlmUserMessage((prompt, ))])
        result: str = response.content.lower()
        try:
            return result.split("choice:")[1]
        except:
            return result
    
    def __apply_augmentation(self, 
                             augmentation: str, 
                             image: InMemoryImage,
                             random_cpu: torch.Generator | None) -> InMemoryImage:
        image_pillow: PIL.Image.Image = image.pillow.convert("RGB")
        if "crop" in augmentation:
            # random_crop
            if random_cpu is not None:
                warnings.warn("In random crop mode, we cannot respect the passed-in random generator "
                              "as torchvision.transforms.RandomCrop does not provide an API for it.")
            
            height: int
            width: int
            width, height = image_pillow.size
            crop_size: int = int(min(height, width) / self.__crop_ratio)
            image_pillow = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(crop_size, crop_size)),
                torchvision.transforms.Resize(size=(height, width), antialias=True)
            ])(image_pillow)
            return InMemoryImage.from_pillow(image_pillow)
        
        if "horizontal" in augmentation:
            # horizontal_flip
            return InMemoryImage.from_pillow(
                torchvision.transforms.functional.hflip(image_pillow)    # pyright: ignore[reportArgumentType]
            )
        
        if "vertical" in augmentation:
            # vertical_flip
            return InMemoryImage.from_pillow(
                torchvision.transforms.functional.vflip(image_pillow)    # pyright: ignore[reportArgumentType]
            )
        
        if "inver" in augmentation:
            # color_inversion
            return InMemoryImage.from_pillow(
                torchvision.transforms.functional.invert(image_pillow)    # pyright: ignore[reportArgumentType]
            )
        
        if "mask" in augmentation:
            # random_mask
            width, height = image_pillow.size
            mask_size: int = int(min(width, height) // self.__mask_ratio)
            left: int = int(torch.randint(0, width - mask_size + 1, [], generator=random_cpu))
            top: int = int(torch.randint(0, height - mask_size + 1, [], generator=random_cpu))

            draw: ImageDraw.ImageDraw = ImageDraw.Draw(image_pillow)
            draw.rectangle([left, top, left + mask_size, top + mask_size], fill=(0, 0, 0))
            
            return InMemoryImage.from_pillow(image_pillow)
        
        # noise
        return InMemoryImage.from_pillow(torchvision.transforms.functional.to_pil_image(
            self.__noise.add_diffusion_noise(
                torchvision.transforms.functional.to_tensor(image_pillow), 
                random_cpu
            )
        ))

    def apply_augmentation(self, 
                           chat_greedy: Callable[[Sequence[LlmMessage]], LlmAssistantMessage], 
                           messages: Sequence[LlmMessage],
                           random: torch.Generator | None) -> Iterable[LlmMessage]:
        message: LlmMessage
        for message in messages:
            if not isinstance(message, LlmUserMessage):
                yield message
                continue
            
            assert len(message.content) == 2
            assert isinstance(message.content[0], InMemoryImage)
            assert isinstance(message.content[1], str)
            query_image: InMemoryImage = message.content[0]
            query_str: str = message.content[1]

            augmentation: str = self.__request_augmentation(chat_greedy, query_str)
            yield LlmUserMessage((self.__apply_augmentation(augmentation, query_image, random), 
                                  query_str))
    
    def __sigmoid_decayed_entropy(self, logits: Tensor2d, gamma: float):
        probabilities: Tensor2d = Tensor2d(torch.nn.functional.softmax(logits, dim=1))
        probabilities_non_zero: Tensor2d = Tensor2d(probabilities.clone())
        probabilities_non_zero[probabilities == 0] = 1
        entropy: Tensor1d = Tensor1d(torch.sum(probabilities * torch.log2(probabilities_non_zero), dim=1))
        entropy = Tensor1d(torch.nn.functional.sigmoid(gamma * entropy))
        return entropy

    def __scaled_normalized_entropy(self, logits: Tensor2d, tau: float):
        probabilities: Tensor2d = Tensor2d(torch.nn.functional.softmax(logits, dim=1))
        probabilities_non_zero: Tensor2d = Tensor2d(probabilities.clone())
        probabilities_non_zero[probabilities == 0] = 1
        entropy: Tensor1d = Tensor1d(-torch.sum(probabilities * torch.log2(probabilities_non_zero), dim=1))
        normalized: Tensor1d = Tensor1d(entropy / math.log2(probabilities.size(1)))
        return Tensor1d(normalized ** (1 / tau))
    
    def __process_logits(self, next_token_logits: Tensor2d, next_token_logits_selfaug: Tensor2d):
        if self.__tau is None:
            beta: Tensor1d = Tensor1d(torch.empty(
                [next_token_logits.size(0)], 
                dtype=next_token_logits.dtype, 
                device=next_token_logits.device).fill_(self.__beta))
        else:
            if self.__tau > 0:
                beta = self.__sigmoid_decayed_entropy(next_token_logits, self.__tau)
            else:
                beta = self.__scaled_normalized_entropy(next_token_logits, -self.__tau)
        # Their original implementations seem to have an error. 
        # We adjusted `cd_config.cd_beta` to `cd_beta`, which may have originated from `cd_tau`.
        threshold: Tensor = torch.log(beta).unsqueeze(-1) + next_token_logits.max(dim=-1, keepdim=True).values

        cd_logits: Tensor = (1 + self.__alpha) * next_token_logits - self.__alpha * next_token_logits_selfaug
        mask: Tensor = next_token_logits < threshold
        return cd_logits.masked_fill(mask, -float("inf"))
    
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
        model_kwargs_selfaug: dict[str, Any],
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
        outputs_selfaug: Any = model._prefill(
            input_ids,
            generation_config,
            model_kwargs_selfaug,
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

                model_inputs_selfaug: dict[str, Any] = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs_selfaug
                )
                with model._optimize_model_for_decode():
                    outputs_selfaug = model_forward(**model_inputs_selfaug, return_dict=True)
            prefill_consumed = True

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            model_kwargs_selfaug = model._update_model_kwargs_for_generation(
                outputs_selfaug,
                model_kwargs_selfaug,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits: Tensor = outputs.logits[:, -1, :].to(copy=True, 
                                                                    dtype=torch.float32, device=input_ids.device)
            next_token_logits_selfaug: Tensor = outputs_selfaug.logits[:, -1, :].to(copy=True, 
                                                                            dtype=torch.float32, device=input_ids.device)
            
            if gate.next(outputs.attentions):
                next_token_logits = self.__process_logits(Tensor2d(next_token_logits), 
                                                          Tensor2d(next_token_logits_selfaug))

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
            outputs_selfaug = None

        if streamer is not None:
            streamer.end()

        return input_ids
    