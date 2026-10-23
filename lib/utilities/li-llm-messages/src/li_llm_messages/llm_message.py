from li_llm_messages.llm_assistant_message import LlmAssistantMessage
from li_llm_messages.llm_system_message import LlmSystemMessage
from li_llm_messages.llm_user_message import LlmUserMessage


type LlmMessage = LlmAssistantMessage | LlmSystemMessage | LlmUserMessage
