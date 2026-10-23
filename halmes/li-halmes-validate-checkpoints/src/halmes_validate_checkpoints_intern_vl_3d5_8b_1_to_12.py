from pathlib import Path

from li_basic_utilities import StaticClass
from li_halmes_configurations import HalmesCommonConfiguration, HalmesFeatureSelectorConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration
from li_halmes_model import HalmesModel
import torch

from li_halmes_validate_checkpoints.main import Main


class Config(StaticClass):
    @staticmethod
    def datasets() -> Path:
        return HalmesOutputPathConfiguration.datasets()
    
    @staticmethod
    def checkpoints() -> Path:
        return HalmesOutputPathConfiguration.checkpoints()

    @staticmethod
    def output() -> Path:
        return HalmesOutputPathConfiguration.best_checkpoints()
    
    @staticmethod
    def target() -> tuple[str, HalmesModel.HalmesFeatureSelector]:
        return (HalmesTargetLlmConfiguration.intern_vl_3d5_8b()["id"],
                HalmesFeatureSelectorConfiguration.intern_vl_3d5_8b_1_12())

    @staticmethod
    def device() -> torch.device:
        return HalmesCommonConfiguration.device()
    

if __name__ == "__main__":
    Main.main(Config.datasets(), Config.checkpoints(), *Config.target(), Config.device(), Config.output())
