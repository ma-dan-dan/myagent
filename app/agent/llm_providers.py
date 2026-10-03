from app.agent.adapter import LiteLLMAdapter


class OpenAILLMAdapter(LiteLLMAdapter):
    PROVIDER = "openai"
    MODEL_PREFIX = ""
    DEFAULT_BASE_URL = None
    API_KEY_ENV = "OPENAI_API_KEY"
    MODEL_ENV = "OPENAI_MODEL"
    BASE_URL_ENV = "OPENAI_BASE_URL"


class DeepSeekLLMAdapter(LiteLLMAdapter):
    PROVIDER = "deepseek"
    MODEL_PREFIX = "deepseek/"
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    MODEL_ENV = "deepseek-flash"
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"


class QwenLLMAdapter(LiteLLMAdapter):
    PROVIDER = "qwen"
    MODEL_PREFIX = "dashscope/"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY_ENV = "QWEN_API_KEY"
    MODEL_ENV = "deepseek-v4.1-flash"
    BASE_URL_ENV = "QWEN_BASE_URL"
