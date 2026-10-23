from pathlib import Path

from li_basic_utilities import StaticClass

from li_halmes_configurations.halmes_target_llm import HalmesTargetLlm


class HalmesTargetLlmConfiguration(StaticClass):
    @staticmethod
    def __llms() -> Path:
        # This is just for convenience; feel free to edit the functions below directly.
        return Path("../../managing/links/llms").absolute()

    @staticmethod
    def qwen3_vl_8b() -> HalmesTargetLlm:
        model_path: Path = HalmesTargetLlmConfiguration.__llms() / "Qwen3-VL-8B-Instruct"

        return {
            "id": "qwen3_vl_8b",
            "model_type": "qwen3_vl",
            "model_path": model_path,

            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50
        }
    
    @staticmethod
    def intern_vl_3d5_8b() -> HalmesTargetLlm:
        model_path: Path = HalmesTargetLlmConfiguration.__llms() / "InternVL3_5-8B-HF"
        
        return {
            "id": "intern_vl_3d5_8b",
            "model_type": "intern_vl_3d5",
            "model_path": model_path,

            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50
        }
