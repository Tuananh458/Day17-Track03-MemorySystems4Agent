from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_PROVIDER_ALIASES = {
    "openai": "openai",
    "custom": "custom",
    "litellm": "custom",
    "antco": "custom",
    "proxy": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
}


def normalize_provider(value: str) -> str:
    key = value.strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider: {value}")
    return _PROVIDER_ALIASES[key]


def _normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            timeout=30,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("CUSTOM_BASE_URL is required for custom/litellm/antco provider")
        if not config.api_key:
            raise ValueError("CUSTOM_API_KEY is required for custom/litellm/antco provider")

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=_normalize_base_url(config.base_url),
            timeout=30,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
