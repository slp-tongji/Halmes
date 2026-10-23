from pathlib import Path

import csfile
from li_basic_utilities import StaticClass
from li_openai import OpenaiLlmParameter, RemoteOpenaiLlm


class HalmesOpenaiLlmConfiguration(StaticClass):
    @staticmethod
    def __secret() -> str:
        # This is just for convenience; feel free to edit the functions below directly.
        return csfile.read_all_text("../../managing/links/secrets/openai")

    @staticmethod
    def __base_url() -> str:
        # This is just for convenience; feel free to edit the functions below directly.
        return "https://sg.uiuiapi.com/v1"

    @staticmethod
    def gpt_4o(parameter: OpenaiLlmParameter) -> RemoteOpenaiLlm:
        return RemoteOpenaiLlm("gpt-4o", 
                               HalmesOpenaiLlmConfiguration.__secret(), 
                               HalmesOpenaiLlmConfiguration.__base_url(),
                               parameter)
    