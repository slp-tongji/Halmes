from pathlib import Path

from li_basic_utilities import StaticClass


class HalmesOutputPathConfiguration(StaticClass):
    @staticmethod
    def __outputs() -> Path:
        # This is just for convenience; feel free to edit the functions below directly.
        path: Path = Path("../../managing/links/outputs").absolute()
        if not path.exists():
            raise RuntimeError("The configured output directory does not exists.\n"
                               "Note that the output directory will not be created automatically "
                               "to prevent it from pointing to an unexpected location.\n"
                               f"The currently configured value is {path}")
        return path / "halmes"

    @staticmethod
    def raw_data() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "raw_data"
    
    @staticmethod
    def labelled_data() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "labelled_data"
    
    @staticmethod
    def datasets() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "datasets"
    
    @staticmethod
    def checkpoints() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "checkpoints"
    
    @staticmethod
    def best_checkpoints() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "best_checkpoints"
    
    @staticmethod
    def inference_outputs() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "inference_outputs"
    
    @staticmethod
    def inference_outputs_scores() -> Path:
        return HalmesOutputPathConfiguration.__outputs() / "inference_outputs_scores"
    