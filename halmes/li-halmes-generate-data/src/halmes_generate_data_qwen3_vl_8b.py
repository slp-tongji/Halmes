from pathlib import Path

from li_data_loader import DatasetInputProvider
from li_halmes_configurations import HalmesDatasetConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration, HalmesCommonConfiguration
import torch
from transformers import StaticCache

from li_halmes_generate_data.llms.qwen_3_vl import Qwen3Vl
from li_halmes_generate_data.main import Main
from li_halmes_generate_data.supported_llm import SupportedLlm


class Config(StaticCache):
    @staticmethod
    def output_directory() -> Path:
        return HalmesOutputPathConfiguration.raw_data()

    @staticmethod
    def target() -> SupportedLlm:
        return Qwen3Vl(HalmesTargetLlmConfiguration.qwen3_vl_8b(),
                       HalmesCommonConfiguration.device())

    @staticmethod
    def model_new_tokens() -> int:
        return 128

    @staticmethod
    def dataset() -> DatasetInputProvider:
        return HalmesDatasetConfiguration.coco_train2014_inputs().slice(0, 5000)

    @staticmethod
    def skip_if_exists() -> bool:
        return True
    
    @staticmethod
    def random_seed() -> str:
        return "8a415209-03f2-4706-91e2-ba767a9e9571"


if __name__ == "__main__":
    Main.main(Config.random_seed(), Config.target(), 
              Config.model_new_tokens(), Config.output_directory(), 
              Config.skip_if_exists(), Config.dataset())
    