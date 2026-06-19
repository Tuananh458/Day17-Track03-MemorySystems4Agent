from __future__ import annotations

from typing import Any

from config import LabConfig, load_config
from langgraph_runtime import (
    AdvancedToolFactory,
    build_advanced_graph,
    delete_thread,
    extract_answer,
    invoke_graph,
    langgraph_available,
)
from llm_chat import llm_dependencies_available
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_candidates, extract_profile_updates
from model_provider import build_chat_model


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline or self.config.force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            decay_half_life_days=self.config.profile_decay_half_life_days,
            min_effective_confidence=self.config.profile_decay_min_confidence,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._current_user_id = "default_user"
        self._current_thread_id = "default_thread"
        self.tool_factory = AdvancedToolFactory(self.profile_store)
        self.langgraph_agent = None
        self.checkpointer = None
        self.chat_model = None
        if not self.force_offline:
            self.langgraph_agent, self.checkpointer, self.chat_model = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langgraph_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def get_profile_text(self, user_id: str) -> str:
        return self.profile_store.get_active_profile_text(user_id)

    def get_full_profile_text(self, user_id: str) -> str:
        return self.profile_store.read_text(user_id)

    def get_structured_entities(self, user_id: str) -> list[dict[str, object]]:
        return self.profile_store.structured_entities_export(user_id)

    def get_compact_context(self, thread_id: str) -> dict[str, object]:
        return self.compact_memory.context(thread_id)

    def get_thread_messages(self, thread_id: str) -> list[dict[str, str]]:
        context = self.compact_memory.context(thread_id)
        messages = context.get("messages", [])
        if isinstance(messages, list):
            return [dict(item) for item in messages if isinstance(item, dict)]
        return []

    def reset_thread(self, thread_id: str) -> None:
        self.compact_memory.state.pop(thread_id, None)
        self.thread_tokens.pop(thread_id, None)
        self.thread_prompt_tokens.pop(thread_id, None)
        delete_thread(self.checkpointer, thread_id)

    def reset_user(self, user_id: str) -> None:
        path = self.profile_store.path_for(user_id)
        if path.exists():
            path.unlink()
        meta_path = self.profile_store.meta_path_for(user_id)
        if meta_path.exists():
            meta_path.unlink()

    def reset_all(self) -> None:
        self.compact_memory.state.clear()
        self.thread_tokens.clear()
        self.thread_prompt_tokens.clear()

    def debug_info(self, user_id: str, thread_id: str) -> dict[str, Any]:
        context = self.compact_memory.context(thread_id)
        return {
            "agent": "advanced",
            "mode": "langgraph-live" if self.langgraph_agent else "offline",
            "message_count": len(context.get("messages", [])),
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "compactions": self.compaction_count(thread_id),
            "memory_file_bytes": self.memory_file_size(user_id),
            "summary_preview": str(context.get("summary", ""))[:500],
        }

    def _memory_context(self) -> dict[str, str]:
        profile = self.profile_store.get_active_profile_text(self._current_user_id).strip() or "(chưa có profile)"
        context = self.compact_memory.context(self._current_thread_id)
        summary = str(context.get("summary", "")).strip() or "(chưa có summary)"
        return {"profile": profile, "summary": summary}

    def _persist_profile_updates(self, user_id: str, message: str) -> dict[str, str]:
        updates: dict[str, str] = {}
        for fact in extract_profile_candidates(message):
            if fact.confidence < self.config.profile_confidence_threshold:
                continue
            self.profile_store.upsert_entity(
                user_id,
                fact.key,
                fact.value,
                confidence=fact.confidence,
                entity_type=fact.entity_type,
            )
            updates[fact.key] = fact.value
        return updates

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        profile_updates = self._persist_profile_updates(user_id, message)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        return self._result(
            answer,
            answer_tokens,
            prompt_tokens,
            profile_updates=profile_updates,
            mode="offline",
        )

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._current_user_id = user_id
        self._current_thread_id = thread_id
        self.tool_factory.user_id = user_id

        profile_updates = self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        try:
            result = invoke_graph(self.langgraph_agent, thread_id, message)
            answer = extract_answer(result)
            if not answer:
                raise ValueError("LangGraph returned empty answer")
        except Exception as exc:
            if self.compact_memory.state.get(thread_id, {}).get("messages"):
                messages = self.compact_memory.state[thread_id]["messages"]  # type: ignore[index]
                if messages and messages[-1].get("role") == "user":
                    messages.pop()
            offline = self._reply_offline(user_id, thread_id, message)
            offline["error"] = str(exc)[:300]
            offline["mode"] = "offline-fallback"
            return offline

        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens

        return self._result(
            answer,
            answer_tokens,
            prompt_tokens,
            profile_updates=profile_updates,
            mode="langgraph-live",
        )

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.get_active_profile_text(user_id)
        context = self.compact_memory.context(thread_id)
        total = estimate_tokens(profile_text)
        total += estimate_tokens(str(context.get("summary", "")))
        for item in context.get("messages", []):
            if isinstance(item, dict):
                total += estimate_tokens(str(item.get("content", "")))
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.active_facts(user_id)
        lowered = message.lower()

        def fact(key: str, default: str = "") -> str:
            return facts.get(key, default)

        parts: list[str] = []

        if any(token in lowered for token in ("tên", "ten g", "ten gi", "ai", "dungct", "bạn biết mình")):
            if fact("name"):
                parts.append(f"Tên bạn là {fact('name')}.")

        if any(token in lowered for token in ("nghề", "nghe", "làm gì", "lam nghe", "làm việc", "profession", "backend", "mlops")):
            if fact("profession"):
                parts.append(f"Nghề nghiệp hiện tại: {fact('profession')}.")

        if any(token in lowered for token in ("ở đâu", "nơi ở", "location", "huế", "đà nẵng", "hà nội")):
            if fact("location"):
                parts.append(f"Hiện tại bạn đang ở {fact('location')}.")

        if any(token in lowered for token in ("style", "trả lời", "bullet", "ưu tiên")):
            if fact("response_style"):
                parts.append(f"Style trả lời bạn thích: {fact('response_style')}.")

        if any(token in lowered for token in ("đồ uống", "uống", "cà phê")):
            if fact("favorite_drink"):
                parts.append(f"Đồ uống yêu thích: {fact('favorite_drink')}.")

        if any(token in lowered for token in ("món ăn", "ăn", "mì")):
            if fact("favorite_food"):
                parts.append(f"Món ăn yêu thích: {fact('favorite_food')}.")

        if any(token in lowered for token in ("nuôi", "corgi", "bơ", "thú cưng", "con gì")):
            if fact("pet"):
                parts.append(f"Bạn nuôi {fact('pet')}.")

        if any(token in lowered for token in ("quan tâm", "mối quan tâm", "python", "tóm tắt")):
            interests = fact("interests")
            if interests:
                parts.append(f"Mối quan tâm chính: {interests}.")

        if parts:
            return " ".join(parts)

        updates = extract_profile_updates(message)
        if updates:
            saved = ", ".join(f"{key}={value}" for key, value in updates.items())
            return (
                f"Đã lưu vào User.md: {saved}. "
                "Advanced sẽ nhớ các fact này kể cả khi bạn mở thread mới."
            )

        if any(token in lowered for token in ("chào", "xin chào", "hello", "hi ")):
            if fact("name"):
                return (
                    f"Chào {fact('name')}! Mình là Advanced Agent. "
                    f"Profile của bạn đang có {len(facts)} fact trong User.md."
                )
            return (
                "Chào bạn! Mình là Advanced Agent. "
                "Hãy giới thiệu tên, nghề, nơi ở — mình sẽ lưu vào User.md để nhớ xuyên thread."
            )

        if facts:
            known = ", ".join(f"{key}={value}" for key, value in sorted(facts.items()))
            return (
                f"Mình đã nhận tin nhắn và vẫn giữ profile: {known}. "
                "Hỏi lại tên/nghề/nơi ở ở thread mới để thấy cross-session recall."
            )

        return (
            "Mình chưa có thông tin dài hạn. "
            "Hãy nói ví dụ: «Mình tên là ..., ở ..., làm ...» để mình ghi vào User.md."
        )

    def _result(
        self,
        answer: str,
        tokens: int,
        prompt_tokens: int,
        profile_updates: dict[str, str] | None = None,
        mode: str = "offline",
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "mode": mode,
            "profile_updates": profile_updates or {},
        }

    def _maybe_build_langchain_agent(self) -> tuple[Any | None, Any | None, Any | None]:
        if not langgraph_available() or not llm_dependencies_available():
            return None, None, None
        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None, None, None
        try:
            model = build_chat_model(self.config.model)
            graph, checkpointer = build_advanced_graph(
                model,
                self.tool_factory,
                self._memory_context,
                compact_threshold_tokens=self.config.compact_threshold_tokens,
                compact_keep_messages=self.config.compact_keep_messages,
            )
            return graph, checkpointer, model
        except Exception:
            return None, None, None
