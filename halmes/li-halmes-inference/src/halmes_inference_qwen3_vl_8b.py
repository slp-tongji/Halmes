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
from li_halmes_inference.models.qwen3_vl.qwen3_vl_avisc import Qwen3VlAvisc
from li_halmes_inference.models.qwen3_vl.qwen3_vl_beam import Qwen3VlBeam
from li_halmes_inference.models.qwen3_vl.qwen3_vl_m3id import Qwen3VlM3id
from li_halmes_inference.models.qwen3_vl.qwen3_vl_selfaug import Qwen3VlSelfaug
from li_halmes_inference.models.qwen3_vl.qwen3_vl_sid import Qwen3VlSid
from li_halmes_inference.models.qwen3_vl.qwen3_vl_vanilla import Qwen3VlVanilla
from li_halmes_inference.models.qwen3_vl.qwen3_vl_vcd import Qwen3VlVcd
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
        return HalmesTargetLlmConfiguration.qwen3_vl_8b()

    @staticmethod
    def features() -> HalmesModel.HalmesFeatureSelector:
        return HalmesFeatureSelectorConfiguration.qwen3_vl_8b_1_36()

    @staticmethod
    def llms() -> Iterable[SupportedLlmProvider]:
        gate_parameter: HalmesGateParameter = HalmesGateParameter()
        device: torch.device = HalmesCommonConfiguration.device()

        always: AlwaysTrueGate = AlwaysTrueGate()
        return [
            lambda target, features, halmes: Qwen3VlVanilla("qwen3_vl_8b_nucleus", target, device),
            lambda target, features, halmes: Qwen3VlBeam("qwen3_vl_8b_beam", target, device),

            lambda target, features, halmes: Qwen3VlVcd("qwen3_vl_8b_vcd", target, device, always),
            lambda target, features, halmes: Qwen3VlM3id("qwen3_vl_8b_m3id", target, device, always),
            lambda target, features, halmes: Qwen3VlAvisc("qwen3_vl_8b_avisc", target, device, always),
            lambda target, features, halmes: Qwen3VlSid("qwen3_vl_8b_sid", target, device, always),
            lambda target, features, halmes: Qwen3VlSelfaug("qwen3_vl_8b_selfaug", target, device, always),

            lambda target, features, halmes: Qwen3VlVcd(
                "qwen3_vl_8b_halmes_vcd", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: Qwen3VlM3id(
                "qwen3_vl_8b_halmes_m3id", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: Qwen3VlAvisc(
                "qwen3_vl_8b_halmes_avisc", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: Qwen3VlSid(
                "qwen3_vl_8b_halmes_sid", target, device, HalmesGate(halmes(device), features, gate_parameter)
            ),
            lambda target, features, halmes: Qwen3VlSelfaug(
                "qwen3_vl_8b_halmes_selfaug", target, device, HalmesGate(halmes(device), features, gate_parameter)
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
