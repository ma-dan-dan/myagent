from app.agent.adapter import LiteLLMAdapter
from app.config import LLM_PROVIDER_CONFIGS


_OPENAI_CONFIG = LLM_PROVIDER_CONFIGS["openai"]
_DEEPSEEK_CONFIG = LLM_PROVIDER_CONFIGS["deepseek"]
_QWEN_CONFIG = LLM_PROVIDER_CONFIGS["qwen"]


class OpenAILLMAdapter(LiteLLMAdapter):
    PROVIDER = _OPENAI_CONFIG.provider
    MODEL_PREFIX = ""
    DEFAULT_BASE_URL = _OPENAI_CONFIG.base_url
    API_KEY_ENV = _OPENAI_CONFIG.api_key_env
    MODEL_ENV = _OPENAI_CONFIG.model


class DeepSeekLLMAdapter(LiteLLMAdapter):
    PROVIDER = _DEEPSEEK_CONFIG.provider
    MODEL_PREFIX = "deepseek/"
    DEFAULT_BASE_URL = _DEEPSEEK_CONFIG.base_url
    API_KEY_ENV = _DEEPSEEK_CONFIG.api_key_env
    MODEL_ENV = _DEEPSEEK_CONFIG.model


class QwenLLMAdapter(LiteLLMAdapter):
    PROVIDER = _QWEN_CONFIG.provider
    MODEL_PREFIX = "dashscope/"
    DEFAULT_BASE_URL = _QWEN_CONFIG.base_url
    API_KEY_ENV = _QWEN_CONFIG.api_key_env
    MODEL_ENV = _QWEN_CONFIG.model
