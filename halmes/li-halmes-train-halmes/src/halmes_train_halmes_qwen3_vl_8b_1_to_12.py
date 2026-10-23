from pathlib import Path

from li_basic_utilities import StaticClass
from li_halmes_configurations import HalmesCommonConfiguration, HalmesFeatureSelectorConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration
from li_halmes_model import HalmesModel
import torch

from li_halmes_train_halmes.main import Main


class Config(StaticClass):
    @staticmethod
    def input() -> Path:
        return HalmesOutputPathConfiguration.datasets()

    @staticmethod
    def output() -> Path:
        return HalmesOutputPathConfiguration.checkpoints()
    
    @staticmethod
    def target() -> str:
        return HalmesTargetLlmConfiguration.qwen3_vl_8b()["id"]

    @staticmethod
    def features() -> HalmesModel.HalmesFeatureSelector:
        return HalmesFeatureSelectorConfiguration.qwen3_vl_8b_1_12()
    
    @staticmethod
    def device() -> torch.device:
        return HalmesCommonConfiguration.device()
    
    @staticmethod
    def random_seed() -> str:
        return "cafe1b32-8a74-46f9-88a5-a38707c6dbad"
    

if __name__ == "__main__":
    Main.main(Config.input(), Config.target(), Config.random_seed(), Config.device(), Config.output(), Config.features())
