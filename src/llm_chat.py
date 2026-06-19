from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

DEFAULT_LLM_TIMEOUT = 15


def llm_dependencies_available() -> bool:
    try:
        import langchain_core  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks).strip()
    return str(content).strip()


def _invoke_chat_inner(model: Any, system_prompt: str, messages: list[dict[str, str]]) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    payload: list[Any] = [SystemMessage(content=system_prompt)]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant":
            payload.append(AIMessage(content=content))
        else:
            payload.append(HumanMessage(content=content))

    response = model.invoke(payload)
    return _extract_text(response)


def invoke_chat(
    model: Any,
    system_prompt: str,
    messages: list[dict[str, str]],
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke_chat_inner, model, system_prompt, messages)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"LLM không phản hồi trong {timeout}s") from exc


def probe_live_model(config) -> tuple[bool, str | None]:
    if config.force_offline:
        return False, "FORCE_OFFLINE=true"
    if not llm_dependencies_available():
        return False, "Thiếu package langchain"
    if not config.model.api_key and config.model.provider != "ollama":
        return False, "Chưa có API key"

    try:
        from model_provider import build_chat_model

        model = build_chat_model(config.model)
        answer = invoke_chat(
            model,
            "Reply with exactly: OK",
            [{"role": "user", "content": "ping"}],
            timeout=12,
        )
        if answer:
            return True, None
        return False, "LLM trả về rỗng"
    except Exception as exc:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            return False, "Gemini hết quota — dùng chế độ offline"
        return False, message[:200]
