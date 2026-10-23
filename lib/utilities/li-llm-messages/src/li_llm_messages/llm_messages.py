from typing import Callable, Iterable

from li_basic_utilities import StaticClass
from li_image_operating import InMemoryImage
from li_logging import TomlLogLike
import tomlkit
import tomlkit.items

from li_llm_messages.llm_assistant_message import LlmAssistantMessage
from li_llm_messages.llm_message import LlmMessage
from li_llm_messages.llm_system_message import LlmSystemMessage
from li_llm_messages.llm_user_message import LlmUserMessage


class LlmMessages(StaticClass):
    @staticmethod
    def get_role(message: LlmMessage, 
                 system: str = "system", 
                 assistant: str = "assistant",
                 user: str = "user"):
        if isinstance(message, LlmSystemMessage):
            return system
        if isinstance(message, LlmAssistantMessage):
            return assistant
        assert isinstance(message, LlmUserMessage)
        return user

    @staticmethod
    def get_content_text(message: LlmMessage, separator: str = "\n"):
        if isinstance(message, LlmUserMessage):
            return separator.join((c for c in message.content if isinstance(c, str)))
        return message.content
    
    @staticmethod
    def modify_images(messages: Iterable[LlmMessage], 
                      modification: Callable[[InMemoryImage], InMemoryImage]):
        result: list[LlmMessage] = []
        message: LlmMessage
        for message in messages:
            if isinstance(message, LlmUserMessage):
                result.append(message.modify_images(modification))
            else:
                result.append(message)
        return result
    
    @staticmethod
    def log_messages(messages: Iterable[LlmMessage], log: TomlLogLike):
        def log_message(table: tomlkit.items.AbstractTable, message: LlmMessage):
            if isinstance(message, LlmSystemMessage):
                table.append("role", "system")
                table.append("content", tomlkit.string(message.content, literal=True, multiline=True))
                return
            if isinstance(message, LlmAssistantMessage):
                table.append("role", "assistant")
                if message.reasoning is not None:
                    table.append("reasoning", tomlkit.string(message.reasoning, literal=True, multiline=True))
                table.append("content", tomlkit.string(message.content, literal=True, multiline=True))
                return

            assert isinstance(message, LlmUserMessage)
            table.append("role", "user")

            content: tomlkit.items.Array = tomlkit.array().multiline(True)
            table.append("content", content)
            block: str | InMemoryImage
            for block in message.content:
                block_table: tomlkit.items.InlineTable = tomlkit.inline_table()
                content.append(block_table)
                if isinstance(block, str):
                    block_table.append("type", "text")
                    block_table.append("text", tomlkit.string(block, literal=True, multiline=True))
                    continue
                block_table.append("type", "image")
                block_table.append("path", log.create_extra_file_in("images", block.mime_bytes))

        with log:
            array: tomlkit.items.AoT = tomlkit.aot()
            message: LlmMessage
            for message in messages:
                table: tomlkit.items.Table = tomlkit.table()
                array.append(table)
                log_message(table, message)
            log.document.append("messages", array)
