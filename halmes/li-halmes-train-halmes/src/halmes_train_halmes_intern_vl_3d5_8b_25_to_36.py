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
        return HalmesTargetLlmConfiguration.intern_vl_3d5_8b()["id"]

    @staticmethod
    def features() -> HalmesModel.HalmesFeatureSelector:
        return HalmesFeatureSelectorConfiguration.intern_vl_3d5_8b_25_36()
    
    @staticmethod
    def device() -> torch.device:
        return HalmesCommonConfiguration.device()
    
    @staticmethod
    def random_seed() -> str:
        return "3ac2a20b-89fc-4c26-bba6-6f8155917d08"
    

if __name__ == "__main__":
    Main.main(Config.input(), Config.target(), Config.random_seed(), Config.device(), Config.output(), Config.features())
