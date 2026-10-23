from pathlib import Path

from li_basic_utilities import StaticClass
from li_halmes_configurations import HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration

from li_halmes_split_datasets.main import Main


class Config(StaticClass):
    @staticmethod
    def input() -> Path:
        return HalmesOutputPathConfiguration.labelled_data()

    @staticmethod
    def output() -> Path:
        return HalmesOutputPathConfiguration.datasets()
    
    @staticmethod
    def target() -> str:
        return HalmesTargetLlmConfiguration.qwen3_vl_8b()["id"]

    @staticmethod
    def random_seed() -> str:
        return "a5285039-2c92-4723-9865-ca0f07d8635f"
    

if __name__ == "__main__":
    Main.main(Config.input(), Config.output(), Config.target(), Config.random_seed())
    