"""Verify Antco AI Gateway connection from .env (does not print API key)."""

from __future__ import annotations

from config import load_config
from llm_chat import invoke_chat, probe_live_model
from model_provider import build_chat_model


def _list_models(config) -> list[str]:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url.rstrip("/") + ("" if config.model.base_url.rstrip("/").endswith("/v1") else "/v1"),
            timeout=15.0,
        )
        models = client.models.list()
        return [item.id for item in models.data[:20]]
    except Exception as exc:
        print(f"models.list failed: {exc}")
        return []


def main() -> None:
    config = load_config()
    print(f"provider={config.model.provider}")
    print(f"model={config.model.model_name}")
    print(f"base_url={config.model.base_url}")

    models = _list_models(config)
    if models:
        print("available_models (first 20):")
        for name in models:
            print(f"  - {name}")

    ok, reason = probe_live_model(config)
    if not ok:
        print(f"FAIL: {reason}")
        return

    model = build_chat_model(config.model)
    answer = invoke_chat(
        model,
        "Tra loi ngan bang tieng Viet.",
        [{"role": "user", "content": "Xin chao, day la test ket noi Antco gateway."}],
        timeout=25,
    )
    print("OK:", answer[:300].encode("utf-8", errors="replace").decode("utf-8"))


if __name__ == "__main__":
    main()
