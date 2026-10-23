from pathlib import Path

from li_basic_utilities import StaticClass
from li_halmes_configurations import HalmesDatasetConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration

from li_halmes_label_data.judges.wrapped_coco_judge import WrappedCocoJudge
from li_halmes_label_data.main import Main
from li_halmes_label_data.supported_judge import SupportedJudge
from li_halmes_label_data.tokenizers.huggingface_tokenizer import HuggingfaceTokenizer


class Config(StaticClass):
    @staticmethod
    def input() -> Path:
        return HalmesOutputPathConfiguration.raw_data()

    @staticmethod
    def output() -> Path:
        return HalmesOutputPathConfiguration.labelled_data()
    
    @staticmethod
    def target() -> HuggingfaceTokenizer:
        return HuggingfaceTokenizer(HalmesTargetLlmConfiguration.qwen3_vl_8b())

    @staticmethod
    def dataset_judge() -> SupportedJudge:
        return WrappedCocoJudge(HalmesDatasetConfiguration.coco_train2014_judge())
    
    @staticmethod
    def multithreading() -> int | None:
        return 4
    

if __name__ == "__main__":
    Main.main(Config.output(), Config.target, Config.input(), 
              Config.multithreading(), Config.dataset_judge)
    