from pathlib import Path

from li_basic_utilities import StaticClass
from li_halmes_configurations import HalmesDatasetConfiguration, HalmesFeatureSelectorConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlmConfiguration

from li_halmes_score_inference_outputs.judges.wrapped_amber_judge import WrappedAmberJudge
from li_halmes_score_inference_outputs.judges.wrapped_bingo_judge import WrappedBingoJudge
from li_halmes_score_inference_outputs.judges.wrapped_coco_judge import WrappedCocoJudge
from li_halmes_score_inference_outputs.judges.wrapped_mmhalbench_judge import WrappedMmhalbenchJudge
from li_halmes_score_inference_outputs.main import Main
from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class Config(StaticClass):
    @staticmethod
    def input() -> Path:
        return HalmesOutputPathConfiguration.inference_outputs()

    @staticmethod
    def output() -> Path:
        return HalmesOutputPathConfiguration.inference_outputs_scores()
    
    @staticmethod
    def target() -> tuple[str, str]:
        return (HalmesTargetLlmConfiguration.intern_vl_3d5_8b()["id"],
                HalmesFeatureSelectorConfiguration.intern_vl_3d5_8b_1_36().id)

    @staticmethod
    def dataset_judge() -> SupportedJudge:
        return SupportedJudge.empty().merge_many([
            WrappedCocoJudge(HalmesDatasetConfiguration.coco_val2014_judge()),
            WrappedAmberJudge(HalmesDatasetConfiguration.amber_judge()),
            WrappedBingoJudge(HalmesDatasetConfiguration.bingo_judge()),
            WrappedMmhalbenchJudge(HalmesDatasetConfiguration.mmhalbench_judge()),
        ])

    @staticmethod
    def skip_if_exists() -> bool:
        return True
    
    @staticmethod
    def multithreading() -> int | None:
        return 4
    

if __name__ == "__main__":
    Main.main(Config.target(), Config.dataset_judge, Config.input(), 
              Config.output(), Config.skip_if_exists(), Config.multithreading())
