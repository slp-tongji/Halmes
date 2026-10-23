from pathlib import Path

from li_amber_input_provider import AmberInputs, AmberInputsGenerative
from li_amber_judge import AmberJudge, AmberJudgeGenerative
from li_basic_utilities import StaticClass
from li_bingo_input_provider import BingoInputs
from li_bingo_judge import BingoJudge
from li_coco_input_provider import CocoInputs
from li_coco_judge import CocoJudge
from li_mmhalbench_input_provider import MmhalbenchInputs
from li_mmhalbench_judge import MmhalbenchJudge
from li_nltk_wrapper import CrossProcessNltkWrapper
from li_openai import OpenaiLlm
import spacy

from li_halmes_configurations.halmes_openai_llm_configuration import HalmesOpenaiLlmConfiguration
from li_halmes_configurations.halmes_tool_configuration import HalmesToolConfiguration


class HalmesDatasetConfiguration(StaticClass):
    @staticmethod
    def __datasets() -> Path:
        # This is just for convenience; feel free to edit the functions below directly.
        return Path("../../managing/links/datasets").absolute()
    
    @staticmethod
    def coco_train2014_inputs() -> CocoInputs:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MS-COCO"
        return CocoInputs(path / "annotations/captions_train2014.json", 
                          path / "train2014")
    
    @staticmethod
    def coco_train2014_judge() -> CocoJudge:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MS-COCO"
        return CocoJudge(path / "annotations/captions_train2014.json",
                         path / "annotations/instances_train2014.json",
                         CrossProcessNltkWrapper(HalmesToolConfiguration.nltk()))
    
    @staticmethod
    def coco_val2014_inputs() -> CocoInputs:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MS-COCO"
        return CocoInputs(path / "annotations/captions_val2014.json", 
                          path / "val2014")
    
    @staticmethod
    def coco_val2014_judge() -> CocoJudge:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MS-COCO"
        return CocoJudge(path / "annotations/captions_val2014.json",
                         path / "annotations/instances_val2014.json",
                         CrossProcessNltkWrapper(HalmesToolConfiguration.nltk()))
    
    @staticmethod
    def amber_inputs() -> AmberInputsGenerative:
        path: Path = HalmesDatasetConfiguration.__datasets() / "AMBER"
        return AmberInputs(path / "image", path / "data").generative
    
    @staticmethod
    def amber_judge() -> AmberJudgeGenerative:
        path: Path = HalmesDatasetConfiguration.__datasets() / "AMBER"
        return AmberJudge(path / "data", 
                          HalmesOpenaiLlmConfiguration.gpt_4o({
                              "prompt_cache_key": "80214b93-ac9b-4fa8-9275-050c9e3d452d",
                              "temperature": 0
                          }),    # don't really need this
                          spacy.load(HalmesToolConfiguration.spacy_encoreweblg()), 
                          CrossProcessNltkWrapper(HalmesToolConfiguration.nltk())).generative
    
    @staticmethod
    def bingo_inputs() -> BingoInputs:
        path: Path = HalmesDatasetConfiguration.__datasets() / "Bingo"
        return BingoInputs(path)
    
    @staticmethod
    def bingo_judge() -> BingoJudge:
        path: Path = HalmesDatasetConfiguration.__datasets() / "Bingo"
        return BingoJudge(BingoInputs(path), 
                          HalmesOpenaiLlmConfiguration.gpt_4o({
                              "prompt_cache_key": "924dd9bb-bd46-4142-85b2-ffa0c3f3415b",
                              "temperature": 0
                          }))
    
    @staticmethod
    def mmhalbench_inputs() -> MmhalbenchInputs:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MMHal-Bench"
        return MmhalbenchInputs(path)
    
    @staticmethod
    def mmhalbench_judge() -> MmhalbenchJudge:
        path: Path = HalmesDatasetConfiguration.__datasets() / "MMHal-Bench"
        return MmhalbenchJudge(MmhalbenchInputs(path), 
                               HalmesOpenaiLlmConfiguration.gpt_4o({
                                   "prompt_cache_key": "3075b6cd-4fc4-4a14-b68f-0f6a0e69d86d",
                                   "temperature": 0
                               }))
    