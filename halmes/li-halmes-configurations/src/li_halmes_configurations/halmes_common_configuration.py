import torch

from li_basic_utilities import StaticClass


class HalmesCommonConfiguration(StaticClass):
    @staticmethod
    def device() -> torch.device:
        return torch.device("cuda", 0)
    