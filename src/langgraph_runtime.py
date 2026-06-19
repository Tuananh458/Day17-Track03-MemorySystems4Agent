from __future__ import annotations

from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

ADVANCED_BASE_SYSTEM = """Bạn là Advanced Agent với 3 lớp memory:
1) short-term memory trong thread (LangGraph + MemorySaver)
2) persistent memory qua User.md (dùng tools read/upsert/edit)
3) compact memory qua SummarizationMiddleware khi hội thoại dài

Quy tắc:
- Dùng tools để đọc/ghi User.md khi cần lưu hoặc recall facts xuyên thread.
- Ưu tiên facts mới nhất khi có correction.
- Trả lời tiếng Việt, ngắn gọn, có ví dụ thực tế khi phù hợp."""


ContextProvider = Callable[[], dict[str, str]]


def langgraph_available() -> bool:
    try:
        from langchain.agents import create_agent  # noqa: F401
        from langgraph.checkpoint.memory import MemorySaver  # noqa: F401

        return True
    except ImportError:
        return False


def extract_answer(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        message_type = getattr(message, "type", "")
        if message_type == "ai":
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def invoke_graph(graph: Any, thread_id: str, message: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)


class ProfileInjectionMiddleware(AgentMiddleware):
    """Inject User.md + compact summary into the system prompt each model call."""

    def __init__(self, context_provider: ContextProvider) -> None:
        super().__init__()
        self.context_provider = context_provider

    def wrap_model_call(self, request, handler):
        context = self.context_provider()
        base = request.system_message.content if request.system_message else ""
        enriched = (
            f"{base}\n\n"
            f"## User.md (persistent profile)\n{context.get('profile', '(empty)')}\n\n"
            f"## Compact summary (older turns)\n{context.get('summary', '(none)')}"
        )
        return handler(request.override(system_message=SystemMessage(content=enriched)))


class AdvancedToolFactory:
    def __init__(self, profile_store) -> None:
        self.profile_store = profile_store
        self.user_id = "default_user"

    def make_tools(self) -> list[Any]:
        store = self.profile_store
        holder = self

        @tool
        def read_user_profile() -> str:
            """Read persistent User.md for the current user (active facts after memory decay)."""
            return store.get_active_profile_text(holder.user_id)

        @tool
        def upsert_user_fact(key: str, value: str) -> str:
            """Save or update one stable fact in User.md."""
            store.upsert_entity(holder.user_id, key.strip(), value.strip(), confidence=0.85)
            return f"Saved {key}={value} to User.md"

        @tool
        def edit_user_profile(search_text: str, replacement: str) -> str:
            """Replace one occurrence inside User.md."""
            changed = store.edit_text(holder.user_id, search_text, replacement)
            return "Updated User.md" if changed else "Text not found in User.md"

        return [read_user_profile, upsert_user_fact, edit_user_profile]


def build_baseline_graph(model: Any, system_prompt: str) -> tuple[Any, MemorySaver]:
    checkpointer = MemorySaver()
    graph = create_agent(
        model,
        tools=None,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return graph, checkpointer


def build_advanced_graph(
    model: Any,
    tool_factory: AdvancedToolFactory,
    context_provider: ContextProvider,
    *,
    compact_threshold_tokens: int,
    compact_keep_messages: int,
) -> tuple[Any, MemorySaver]:
    checkpointer = MemorySaver()
    middleware = [
        ProfileInjectionMiddleware(context_provider),
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", compact_threshold_tokens),
            keep=("messages", compact_keep_messages),
        ),
    ]
    graph = create_agent(
        model,
        tools=tool_factory.make_tools(),
        system_prompt=ADVANCED_BASE_SYSTEM,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    return graph, checkpointer


def delete_thread(checkpointer: MemorySaver | None, thread_id: str) -> None:
    if checkpointer is None:
        return
    try:
        checkpointer.delete_thread(thread_id)
    except Exception:
        return
