from typing import Literal

from li_basic_utilities import StaticClass

from li_halmes_model import HalmesModel


class HalmesFeatureSelectorConfiguration(StaticClass):
    @staticmethod
    def qwen3_vl_8b_1_36() -> HalmesModel.HalmesFeatureSelector:
        return HalmesModel.HalmesFeatureSelector("1-36", True, True, True, True, True, [[True] * 32] * 36)
    
    @staticmethod
    def qwen3_vl_8b_1_12() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("1-12", True, True, True, True, True, 
                                                 [true_layer] * 12 + [false_layer] * 24)
    
    @staticmethod
    def qwen3_vl_8b_13_24() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("13-24", True, True, True, True, True, 
                                                 [false_layer] * 12 + [true_layer] * 12 + [false_layer] * 12)
    
    @staticmethod
    def qwen3_vl_8b_25_36() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("25-36", True, True, True, True, True, 
                                                 [false_layer] * 24 + [true_layer] * 12)
    
    @staticmethod
    def intern_vl_3d5_8b_1_36() -> HalmesModel.HalmesFeatureSelector:
        return HalmesModel.HalmesFeatureSelector("1-36", True, True, True, True, True, [[True] * 32] * 36)
    
    @staticmethod
    def intern_vl_3d5_8b_1_12() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("1-12", True, True, True, True, True, 
                                                 [true_layer] * 12 + [false_layer] * 24)
    
    @staticmethod
    def intern_vl_3d5_8b_13_24() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("13-24", True, True, True, True, True, 
                                                 [false_layer] * 12 + [true_layer] * 12 + [false_layer] * 12)
    
    @staticmethod
    def intern_vl_3d5_8b_25_36() -> HalmesModel.HalmesFeatureSelector:
        true_layer: list[Literal[True]] = [True] * 32
        false_layer: list[Literal[False]] = [False] * 32
        return HalmesModel.HalmesFeatureSelector("25-36", True, True, True, True, True, 
                                                 [false_layer] * 24 + [true_layer] * 12)
    