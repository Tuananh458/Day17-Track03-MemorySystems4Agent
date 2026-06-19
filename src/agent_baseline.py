from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from langgraph_runtime import (
    build_baseline_graph,
    delete_thread,
    extract_answer,
    invoke_graph,
    langgraph_available,
)
from llm_chat import invoke_chat, llm_dependencies_available
from memory_store import estimate_tokens
from model_provider import build_chat_model


BASELINE_SYSTEM = """Bạn là Baseline Agent — chỉ có short-term memory trong cùng một thread hội thoại (LangGraph MemorySaver).

Quy tắc:
- Chỉ dựa vào các tin nhắn trong thread hiện tại.
- KHÔNG giả vờ nhớ thông tin từ thread hoặc phiên khác.
- Nếu người dùng hỏi fact chưa xuất hiện trong thread này, hãy nói rõ bạn không nhớ.
- Trả lời bằng tiếng Việt, ngắn gọn và tự nhiên."""


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline or self.config.force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langgraph_agent = None
        self.checkpointer = None
        self.chat_model = None
        if not self.force_offline:
            self.langgraph_agent, self.checkpointer, self.chat_model = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id
        if self.langgraph_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        del thread_id
        return 0

    def memory_file_size(self, user_id: str) -> int:
        del user_id
        return 0

    def get_thread_messages(self, thread_id: str) -> list[dict[str, str]]:
        return list(self._session(thread_id).messages)

    def reset_thread(self, thread_id: str) -> None:
        self.sessions.pop(thread_id, None)
        delete_thread(self.checkpointer, thread_id)

    def reset_all(self) -> None:
        self.sessions.clear()

    def debug_info(self, thread_id: str) -> dict[str, Any]:
        session = self.sessions.get(thread_id, SessionState())
        return {
            "agent": "baseline",
            "mode": "langgraph-live" if self.langgraph_agent else "offline",
            "message_count": len(session.messages),
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "compactions": 0,
            "memory_file_bytes": 0,
        }

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _find_in_thread(self, session: SessionState, patterns: list[str]) -> str | None:
        for message in session.messages:
            if message.get("role") != "user":
                continue
            text = message.get("content", "")
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    def _offline_answer(self, session: SessionState, message: str) -> str:
        lowered = message.lower()

        if any(token in lowered for token in ("chào", "xin chào", "hello", "hi ")):
            return (
                "Chào bạn! Mình là Baseline Agent. "
                "Mình chỉ nhớ tin nhắn trong thread hiện tại — sang thread mới là quên."
            )

        if any(token in lowered for token in ("tên", "tên gì", "mình là ai", "bạn biết mình")):
            name = self._find_in_thread(
                session,
                [r"mình tên là\s+([^.,;!\n]+)", r"tên mình là\s+([^.,;!\n]+)", r"tên là\s+([^.,;!\n]+)"],
            )
            if name:
                return f"Trong thread này, bạn đã giới thiệu tên là {name}."
            return "Trong thread này mình chưa thấy bạn nói tên. Mình không nhớ thread khác."

        if any(token in lowered for token in ("nghề", "làm gì", "làm việc", "mlops", "backend")):
            job = self._find_in_thread(session, [r"làm\s+([^.,;!\n]+)"])
            if job:
                return f"Trong thread này bạn nói làm {job}."
            return "Trong thread này mình chưa thấy bạn nói về nghề nghiệp."

        if any(token in lowered for token in ("ở đâu", "nơi ở", "đang ở")):
            location = self._find_in_thread(
                session,
                [r"mình ở\s+([^.,;!\n]+)", r"đang ở\s+([^.,;!\n]+)", r"hiện tại mình đang ở\s+([^.,;!\n]+)"],
            )
            if location:
                return f"Trong thread này bạn nói đang ở {location}."
            return "Trong thread này mình chưa thấy bạn nói nơi ở."

        if any(token in lowered for token in ("đồ uống", "uống", "cà phê")):
            drink = self._find_in_thread(session, [r"đồ uống yêu thích(?:\s+là)?\s+([^.,;!\n]+)", r"uống\s+([^.,;!\n]+)"])
            if drink:
                return f"Trong thread này bạn thích {drink}."
            return "Trong thread này mình chưa thấy bạn nói đồ uống yêu thích."

        user_turns = sum(1 for item in session.messages if item.get("role") == "user")
        return (
            f"Đã nhận tin nhắn của bạn. "
            f"Baseline đang giữ {user_turns} lượt user trong thread này. "
            f"Hãy thử hỏi lại tên/nghề/nơi ở sau khi đã giới thiệu trong cùng thread."
        )

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        context_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in session.messages
        )
        session.prompt_tokens_processed += estimate_tokens(context_text)

        answer = self._offline_answer(session, message)
        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens

        return self._result(answer, answer_tokens, session.prompt_tokens_processed, mode="offline")

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(BASELINE_SYSTEM) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens

        try:
            result = invoke_graph(self.langgraph_agent, thread_id, message)
            answer = extract_answer(result)
            if not answer:
                raise ValueError("LangGraph returned empty answer")
        except Exception as exc:
            session.messages.pop()
            offline = self._reply_offline(thread_id, message)
            offline["error"] = str(exc)[:300]
            offline["mode"] = "offline-fallback"
            return offline

        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        return self._result(answer, answer_tokens, session.prompt_tokens_processed, mode="langgraph-live")

    def _result(self, answer: str, tokens: int, prompt_tokens: int, mode: str) -> dict[str, Any]:
        return {
            "answer": answer,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "mode": mode,
        }

    def _maybe_build_langchain_agent(self) -> tuple[Any | None, Any | None, Any | None]:
        if not langgraph_available() or not llm_dependencies_available():
            return None, None, None
        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None, None, None
        try:
            model = build_chat_model(self.config.model)
            graph, checkpointer = build_baseline_graph(model, BASELINE_SYSTEM)
            return graph, checkpointer, model
        except Exception:
            return None, None, None
