import abc
import time
from typing import Callable, Iterable, Sequence

import instructor
from li_image_operating.in_memory_image import InMemoryImage
from li_llm_messages.llm_assistant_message import LlmAssistantMessage
from li_llm_messages.llm_message import LlmMessage
from li_llm_messages.llm_messages import LlmMessages
from li_llm_messages.llm_system_message import LlmSystemMessage
from li_logging.toml_log_like import TomlLogLike
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionContentPartParam, ChatCompletionMessageParam, ChatCompletionMessageToolCallUnion, ChatCompletionMessageFunctionToolCall
import pydantic

from li_openai.openai_llm_parameter import OpenaiLlmParameter


class OpenaiLlm(abc.ABC):
    def __init__(self,
                 model: str,
                 api_key: str, 
                 base_url: str,
                 retry: Sequence[float],
                 retry_mapping: bool,
                 retry_structured: int) -> None:
        super().__init__()

        self.__retry = tuple(retry)
        self.__retry_mapping = retry_mapping
        self.__retry_structured = retry_structured
        
        self.__model = model
        self.__client = OpenAI(api_key=api_key, base_url=base_url)

    def _retry(self):
        return self.__retry
    
    def _do_with_retry[T](self,
                          callable: Callable[[], T]):
        delay: float
        for delay in self._retry():
            try:
                return callable()
            except:
                time.sleep(delay)
                pass
        return callable()

    @staticmethod
    def map_message_to_itself(message: LlmAssistantMessage):
        return message
    
    def chat[T](self,
                history: Iterable[LlmMessage], 
                log: TomlLogLike,
                mapping: Callable[[LlmAssistantMessage], T] = map_message_to_itself) -> T:
        result: T
        result, history = self.chat_non_log(history, mapping)
        LlmMessages.log_messages(history, log)
        return result
    
    def chat_structured[T: pydantic.BaseModel](self,
                                               history: Iterable[LlmMessage], 
                                               structure: type[T],
                                               log: TomlLogLike) -> T:
        result: T
        result, history = self.chat_structured_non_log(history, structure)
        LlmMessages.log_messages(history, log)
        return result

    def _convert_response(self, response: str) -> LlmAssistantMessage:
        return LlmAssistantMessage(response, None)

    def chat_non_log[T](
            self,
            history: Iterable[LlmMessage], 
            mapping: Callable[[LlmAssistantMessage], T] = map_message_to_itself
        ) -> tuple[T, list[LlmMessage]]:
        history = list(history)

        messages: list = self.__to_openai_messages(history)
        def chat():
            response: ChatCompletion = self.__client.chat.completions.create(
                model=self.__model,
                messages=messages,
                **self._get_parameter()
            )
            assert response.choices[0].message.content is not None
            return self._convert_response(response.choices[0].message.content)
        
        if self.__retry_mapping:
            def chat_and_map():
                response: LlmAssistantMessage = chat()
                mapped: T = mapping(response)
                return response, mapped
            response: LlmAssistantMessage
            result: T
            response, result = self._do_with_retry(chat_and_map)
        else:
            response = self._do_with_retry(chat)
            result = mapping(response)
        
        history.append(response)
        return result, history
    
    def chat_structured_non_log[T: pydantic.BaseModel](self,
                                                       history: Iterable[LlmMessage], 
                                                       structure: type[T]) -> tuple[T, list[LlmMessage]]:
        history = list(history)

        messages: list = self.__to_openai_messages(history)
        instructor_client: instructor.Instructor = instructor.from_openai(self.__client)
        def chat():
            result: T = instructor_client.chat.completions.create(
                response_model=structure,
                max_retries=self.__retry_structured,
                model=self.__model,
                messages=messages,
                **self._get_parameter()
            )
            raw: ChatCompletion = getattr(result, "_raw_response")
            calls: list[ChatCompletionMessageToolCallUnion] | None = raw.choices[0].message.tool_calls
            assert calls is not None
            assert isinstance(calls[0], ChatCompletionMessageFunctionToolCall)
            return result, LlmAssistantMessage(calls[0].function.arguments, None)
        
        result: tuple[T, LlmAssistantMessage] = self._do_with_retry(chat)
        history.append(result[1])
        return result[0], history

    def __to_openai_messages(self, messages: Iterable[LlmMessage]):
        result: list[ChatCompletionMessageParam] = []
        message: LlmMessage
        for message in messages:
            if isinstance(message, LlmSystemMessage):
                result.append({
                    "role": "system",
                    "content": message.content
                })
            if isinstance(message, LlmAssistantMessage):
                result.append({
                    "role": "assistant",
                    "content": message.content
                })

            content: list[ChatCompletionContentPartParam] = []
            block: str | InMemoryImage
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                    continue
                content.append({
                    "type": "image_url", 
                    "image_url": { "url": block.data_url }
                })
            result.append({"role": "user", "content": content})

        return result

    @abc.abstractmethod
    def _get_parameter(self) -> OpenaiLlmParameter:
        ...
