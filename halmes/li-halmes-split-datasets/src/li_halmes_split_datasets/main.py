import math
from pathlib import Path
from random import Random
from typing import Iterable

import csdir
from li_basic_utilities import StaticClass
from li_halmes_data_preparer import HalmesDataManager, HalmesDatasetItem
from statictorch import Tensor2d
import torch


class Main(StaticClass):
    @staticmethod
    def __split(keys: Iterable[str], random: Random, ratio: float = 0.8):
        keys = list(keys)
        random.shuffle(keys)

        position: int = round(len(keys) * ratio)
        assert 0 < position < len(keys)
        return keys[:position], keys[position:]
    
    @staticmethod
    def __merge(keys1: Iterable[str], 
                keys2: Iterable[str], 
                random: Random, 
                under_sample: float | None = None):
        keys1 = list(keys1)

        if under_sample is not None:
            keys2 = list(keys2)
            if len(keys1) / len(keys2) > under_sample:
                expected_length: int = math.ceil(under_sample * len(keys2))
                if len(keys1) > expected_length:
                    keys1 = random.sample(keys1, expected_length)
            else:
                expected_length = math.floor(len(keys1) / under_sample)
                if len(keys2) > expected_length:
                    keys2 = random.sample(keys2, expected_length)

        keys1.extend(keys2)
        random.shuffle(keys1)
        return keys1
    
    @staticmethod
    def __get_values(keys: Iterable[str], 
                     hallucination: HalmesDataManager[Tensor2d],
                     non_hallucination: HalmesDataManager[Tensor2d]) -> Iterable[HalmesDatasetItem]:
        key: str
        for key in keys:
            value: Tensor2d | None
            _, value = hallucination.get(key)
            
            if value is not None:
                yield {
                    "feature": value,
                    "is_hallucination": True
                }
            else:
                _, value = non_hallucination.get(key)
                assert value is not None
                yield {
                    "feature": value,
                    "is_hallucination": False
                }

    @staticmethod
    def main(input: Path, 
             output: Path, 
             target_id: str,
             random_seed: str) -> None:
        input = input / target_id
        output = csdir.create_directory(output / target_id)
        print(f"Output: {output}")

        random: Random = Random(random_seed)

        hallucination: HalmesDataManager[Tensor2d] = HalmesDataManager(input / "hallucination", True)
        non_hallucination: HalmesDataManager[Tensor2d] = HalmesDataManager(input / "non_hallucination", True)

        train: list[str]
        validation: list[str]
        train, validation = Main.__split(hallucination.find(), random)
        
        train_non_hallucination: list[str]
        validation_non_hallucination: list[str]
        train_non_hallucination, validation_non_hallucination = Main.__split(
            non_hallucination.find(), random)
        
        train = Main.__merge(train, train_non_hallucination, random)
        validation = Main.__merge(validation, validation_non_hallucination, random, 1)

        torch.save(list(Main.__get_values(train, hallucination, non_hallucination)), output / "train.pt")
        torch.save(list(Main.__get_values(validation, hallucination, non_hallucination)), output / "validation.pt")

        print("Done.")