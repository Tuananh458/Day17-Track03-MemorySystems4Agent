from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    profile_confidence_threshold: float
    model: ProviderConfig
    judge_model: ProviderConfig
    force_offline: bool = False


_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gemini-3-flash",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _provider_credentials(provider: str) -> tuple[str | None, str | None]:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY"), None
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY"), None
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY"), None
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY"), None
    if provider == "custom":
        return os.getenv("CUSTOM_API_KEY"), os.getenv("CUSTOM_BASE_URL")
    if provider == "ollama":
        return None, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    raise ValueError(f"Unsupported provider: {provider}")


def _provider_from_env(prefix: str) -> ProviderConfig:
    raw_provider = os.getenv(f"{prefix}_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
    provider = normalize_provider(raw_provider)
    model_name = os.getenv(
        f"{prefix}_MODEL",
        _DEFAULT_MODELS.get(provider, "gpt-4o-mini"),
    )
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0.2" if prefix == "LLM" else "0.0"))
    api_key, base_url = _provider_credentials(provider)
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        profile_confidence_threshold=float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.7")),
        model=_provider_from_env("LLM"),
        judge_model=_provider_from_env("JUDGE"),
        force_offline=_env_bool("FORCE_OFFLINE", default=False),
    )
