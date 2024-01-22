import json
import random

from enum import Enum
from langchain.chat_models import AzureChatOpenAI


class LLMNames(str, Enum):
    gpt_35_turbo = 'gpt-35-turbo'
    gpt_35_turbo_16k = 'gpt-35-turbo-16k'
    gpt_4 = 'gpt-4'
    gpt_4_32k = 'gpt-4-32k'
    gpt_4_turbo = 'gpt-4-turbo'
    gpt_4_turbo_preview = 'gpt-4-1106-preview'


def get_chat_azure_openai_client(
        api_key: str, endpoint: str, deployment_name: str,
        temperature: float = 0.0, max_tokens: int = 8192,
        streaming: bool = False, callbacks: list = None,
):
    """
    :param api_key: Azure OpenAI API Key
    :param endpoint: Azure OpenAI endpoint
    :param deployment_name: Azure OpenAI deployment name
    :param temperature: 0.0 ~ 1.0
    :param max_tokens: See https://www.scriptbyai.com/token-limit-openai-chatgpt
    :param streaming: Streaming mode if True
    :param callbacks:
    :return: AzureOpenAI Client

    See tests/llm_healthcheck/test_azure_openai.py for example usage.
    """
    return AzureChatOpenAI(
        openai_api_base=endpoint,
        openai_api_version="2023-07-01-preview",
        deployment_name=deployment_name,
        openai_api_key=api_key,
        openai_api_type="azure",
        temperature=temperature,
        max_tokens=max_tokens,  # gpt-4-32k (max 32768)
        streaming=streaming,
        callbacks=callbacks,
    )