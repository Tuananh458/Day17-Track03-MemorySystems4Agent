from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import format_rows, run_agent_benchmark
from config import load_config

from llm_chat import probe_live_model

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Day 17 Memory Agents Demo", version="1.0.0")
config = load_config()

llm_live, llm_reason = probe_live_model(config)
use_offline = not llm_live
if use_offline:
    print(f"[demo] OFFLINE mode -- {llm_reason}")

baseline_agent = BaselineAgent(config, force_offline=use_offline)
advanced_agent = AdvancedAgent(config, force_offline=use_offline)


class ChatRequest(BaseModel):
    user_id: str = Field(default="demo_user", min_length=1, max_length=64)
    thread_id: str = Field(default="thread-1", min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class ResetRequest(BaseModel):
    user_id: str = Field(default="demo_user", min_length=1, max_length=64)
    thread_id: str | None = None
    clear_profile: bool = False


def _agent_payload(agent_name: str, agent, user_id: str, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    debug = (
        advanced_agent.debug_info(user_id, thread_id)
        if agent_name == "advanced"
        else baseline_agent.debug_info(thread_id)
    )
    return {
        "answer": result.get("answer", ""),
        "tokens": result.get("tokens", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "mode": result.get("mode", "offline"),
        "profile_updates": result.get("profile_updates", {}),
        "error": result.get("error"),
        "stats": debug,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": config.model.provider,
        "model": config.model.model_name,
        "base_url": config.model.base_url,
        "force_offline": config.force_offline,
        "llm_live": llm_live,
        "llm_reason": llm_reason,
        "baseline_mode": baseline_agent.debug_info("health")["mode"],
        "advanced_mode": advanced_agent.debug_info("demo_user", "health")["mode"],
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    baseline_result = baseline_agent.reply(payload.user_id, payload.thread_id, message)
    advanced_result = advanced_agent.reply(payload.user_id, payload.thread_id, message)

    compact_context = advanced_agent.get_compact_context(payload.thread_id)
    return {
        "baseline": _agent_payload("baseline", baseline_agent, payload.user_id, payload.thread_id, baseline_result),
        "advanced": _agent_payload("advanced", advanced_agent, payload.user_id, payload.thread_id, advanced_result),
        "profile_md": advanced_agent.get_profile_text(payload.user_id),
        "compact_summary": str(compact_context.get("summary", "")),
        "compactions": advanced_agent.compaction_count(payload.thread_id),
    }


@app.post("/api/new-thread")
def new_thread() -> dict[str, str]:
    return {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}


@app.post("/api/reset")
def reset(payload: ResetRequest) -> dict[str, str]:
    if payload.thread_id:
        baseline_agent.reset_thread(payload.thread_id)
        advanced_agent.reset_thread(payload.thread_id)
    else:
        baseline_agent.reset_all()
        advanced_agent.reset_all()

    if payload.clear_profile:
        advanced_agent.reset_user(payload.user_id)

    return {"status": "ok"}


@app.get("/api/memory/{user_id}")
def memory(user_id: str, thread_id: str = "thread-1") -> dict[str, Any]:
    compact_context = advanced_agent.get_compact_context(thread_id)
    return {
        "profile_md": advanced_agent.get_profile_text(user_id),
        "profile_entities": advanced_agent.get_structured_entities(user_id),
        "memory_file_bytes": advanced_agent.memory_file_size(user_id),
        "compact_summary": str(compact_context.get("summary", "")),
        "compactions": advanced_agent.compaction_count(thread_id),
        "baseline": baseline_agent.debug_info(thread_id),
        "advanced": advanced_agent.debug_info(user_id, thread_id),
    }


@app.get("/api/benchmark/quick")
def quick_benchmark() -> dict[str, Any]:
    conversations = [
        {
            "id": "demo-quick",
            "user_id": "demo_quick",
            "turns": [
                "Mình tên là DũngCT, ở Huế, làm MLOps engineer.",
                "Mình thích cà phê sữa đá và trả lời ngắn gọn.",
            ],
            "recall_questions": [
                {
                    "question": "Mình tên gì và làm nghề gì?",
                    "expected_contains": ["DũngCT", "MLOps engineer"],
                }
            ],
        }
    ]

    offline_config = load_config(config.base_dir)
    offline_baseline = BaselineAgent(offline_config, force_offline=True)
    offline_advanced = AdvancedAgent(offline_config, force_offline=True)

    rows = [
        run_agent_benchmark("Baseline", offline_baseline, conversations, offline_config),
        run_agent_benchmark("Advanced", offline_advanced, conversations, offline_config),
    ]
    return {"table": format_rows(rows), "rows": [row.__dict__ for row in rows]}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "demo_server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
