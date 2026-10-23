from pathlib import Path
from typing import Iterable

from li_basic_utilities import StaticClass
from li_data_loader import DatasetInputProvider
from li_halmes_configurations import HalmesCommonConfiguration, HalmesDatasetConfiguration, HalmesFeatureSelectorConfiguration, HalmesOutputPathConfiguration, HalmesTargetLlm, HalmesTargetLlmConfiguration
from li_halmes_model import HalmesGateParameter, HalmesModel
import torch

from li_halmes_inference.gates.always_true_gate import AlwaysTrueGate
from li_halmes_inference.gates.halmes_gate import HalmesGate
from li_halmes_inference.main import Main
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_avisc import InternVl3d5Avisc
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_beam import InternVl3d5Beam
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_m3id import InternVl3d5M3id
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_selfaug import InternVl3d5Selfaug
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_sid import InternVl3d5Sid
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_vanilla import InternVl3d5Vanilla
from li_halmes_inference.models.intern_vl_3d5.intern_vl_3d5_vcd import InternVl3d5Vcd
from li_halmes_inference.supported_llm_provider import SupportedLlmProvider


class Config(StaticClass):
    @staticmethod
    def checkpoints() -> Path:
        return HalmesOutputPathConfiguration.best_checkpoints()

    @staticmethod
    def inference_outputs() -> Path:
        return HalmesOutputPathConfiguration.inference_outputs()
    
    @staticmethod
    def target() -> HalmesTargetLlm:
        return HalmesTargetLlmConfiguration.intern_vl_3d5_8b()

    @staticmethod
    def features() -> HalmesModel.HalmesFeatureSelector:
        return HalmesFeatureSelectorConfiguration.intern_vl_3d5_8b_1_36()

    @staticmethod
    def llms() -> Iterable[SupportedLlmProvider]:
        gate_parameter: HalmesGateParameter = HalmesGateParameter(theta_high=0.75, theta_low=0.65, cool_down=6)
        device: torch.device = HalmesCommonConfiguration.device()

        always: AlwaysTrueGate = AlwaysTrueGate()
        return [
            lambda target, features, halmes: InternVl3d5Vanilla("intern_vl_3d5_8b_nucleus", target, device),
            lambda target, features, halmes: InternVl3d5Beam("intern_vl_3d5_8b_beam", target, device),

            lambda target, features, halmes: InternVl3d5Vcd("intern_vl_3d5_8b_vcd", target, device, always),
            lambda target, features, halmes: InternVl3d5M3id("intern_vl_3d5_8b_m3id", target, device, always),
            lambda target, features, halmes: InternVl3d5Avisc("intern_vl_3d5_8b_avisc", target, device, always),
            lambda target, features, halmes: InternVl3d5Sid("intern_vl_3d5_8b_sid", target, device, always),
            lambda target, features, halmes: InternVl3d5Selfaug("intern_vl_3d5_8b_selfaug", target, device, always),

            lambda target, features, halmes: InternVl3d5Vcd(
                "intern_vl_3d5_8b_halmes_vcd", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: InternVl3d5M3id(
                "intern_vl_3d5_8b_halmes_m3id", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: InternVl3d5Avisc(
                "intern_vl_3d5_8b_halmes_avisc", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: InternVl3d5Sid(
                "intern_vl_3d5_8b_halmes_sid", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: InternVl3d5Selfaug(
                "intern_vl_3d5_8b_halmes_selfaug", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
        ]

    @staticmethod
    def model_new_tokens() -> int:
        return 128

    @staticmethod
    def dataset() -> DatasetInputProvider:
        return DatasetInputProvider.empty().merge_many([
            HalmesDatasetConfiguration.coco_val2014_inputs().slice(0, 500),
            HalmesDatasetConfiguration.amber_inputs(),
            HalmesDatasetConfiguration.bingo_inputs(),
            HalmesDatasetConfiguration.mmhalbench_inputs(),
        ])

    @staticmethod
    def skip_if_exists() -> bool:
        return True
    

if __name__ == "__main__":
    Main.main(Config.target(), Config.features(), Config.llms(), 
              Config.dataset(), Config.checkpoints(), Config.inference_outputs(), 
              Config.skip_if_exists(), Config.model_new_tokens())
