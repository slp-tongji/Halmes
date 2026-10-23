import textwrap
from typing import NamedTuple


class LlmSystemMessage(NamedTuple):
    content: str

    def dedented(self):
        return LlmSystemMessage(textwrap.dedent(self.content).strip("\r\n"))
