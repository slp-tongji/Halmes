import math
from typing import Sequence
import typing

from li_basic_utilities import StaticClass
from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d
import torch

from li_halmes_model.halmes_feature import HalmesFeature


class HalmesFeatureExtractor(StaticClass):
    @staticmethod
    def __extract(
        attention_of_one_step: Sequence[Tensor3d],
        image_mask: Tensor1d,
        text_mask: Tensor1d
    ) -> HalmesFeature:
        assert attention_of_one_step[0].device == image_mask.device

        visual_attentions: list[Tensor1d] = []
        text_attentions: list[Tensor0d] = []
        generated_attentions: list[Tensor0d] = []
        entropies: list[Tensor0d] = []

        attention_3d: Tensor3d
        for attention_3d in attention_of_one_step:
            attention: Tensor2d = Tensor2d(attention_3d[:, -1, :].to(torch.float32))
            
            attention_prompt: Tensor2d = Tensor2d(attention[:, :image_mask.size(0)])
            attention_generated: Tensor2d = Tensor2d(attention[:, image_mask.size(0):])

            visual_attention: Tensor1d = Tensor1d(attention_prompt[:, image_mask].sum(dim=-1))
            visual_attentions.append(visual_attention)

            text_attentions.append(Tensor0d(attention_prompt[:, text_mask].mean()))
            generated_attentions.append(Tensor0d(attention_generated.mean().nan_to_num(0)))

            entropy: Tensor1d = Tensor1d(-(attention * attention.clamp(min=1e-10).log()).sum(dim=-1))
            entropy = Tensor1d(entropy / math.log(attention.size(-1)))
            entropies.append(Tensor0d(entropy.mean()))

        var: Tensor2d = Tensor2d(torch.stack(typing.cast(list[torch.Tensor], visual_attentions), dim=0))
        var_mean: Tensor1d = Tensor1d(var.mean(1))
        attention_generated_text: Tensor0d = Tensor0d(
            torch.stack(typing.cast(list[torch.Tensor], generated_attentions), dim=0).mean()
        )

        return {
            "var": var,
            "svar": Tensor0d(var_mean.sum()),
            "attention_text": Tensor0d(
                torch.stack(typing.cast(list[torch.Tensor], text_attentions), dim=0).mean()
            ),
            "attention_generated_text": attention_generated_text,
            "text_visual_ratio": Tensor0d(attention_generated_text / var_mean.mean()),
            "outgoing_entropy": Tensor0d(
                torch.stack(typing.cast(list[torch.Tensor], entropies), dim=0).mean()
            ),
        }
    
    
    @staticmethod
    def extract(
        attentions: Sequence[Sequence[Tensor3d]],
        mm_token_type_ids: Tensor1d
    ) :
        image_mask: Tensor1d = Tensor1d(mm_token_type_ids == 1)
        text_mask: Tensor1d = Tensor1d(mm_token_type_ids == 0)
        attention: Sequence[Tensor3d]
        for attention in attentions:
            yield HalmesFeatureExtractor.__extract(
                attention, Tensor1d(image_mask), Tensor1d(text_mask)
            )

    @staticmethod
    def extract_single(
        attention: Sequence[Tensor3d],
        mm_token_type_ids: Tensor1d
    ) :
        image_mask: Tensor1d = Tensor1d(mm_token_type_ids == 1)
        text_mask: Tensor1d = Tensor1d(mm_token_type_ids == 0)
        return HalmesFeatureExtractor.__extract(
            attention, Tensor1d(image_mask), Tensor1d(text_mask)
        )

    @staticmethod
    def to_single_tensor(feature: HalmesFeature):
        return Tensor1d(torch.cat([
            feature["var"].flatten(),
            torch.stack([
                feature["svar"],
                feature["attention_text"],
                feature["attention_generated_text"],
                feature["text_visual_ratio"],
                feature["outgoing_entropy"],
            ], dim=0)
        ], dim=0))
    