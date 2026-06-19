from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for item in expected if item.lower() in answer_lower)
    return found / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    length_bonus = 0.2 if 20 <= len(answer) <= 400 else 0.0
    return min(1.0, recall + length_bonus)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    thread_agent_tokens: dict[str, int] = {}
    thread_prompt_tokens: dict[str, int] = {}
    total_recall = 0.0
    total_quality = 0.0
    recall_count = 0
    total_memory_growth = 0
    max_compactions = 0

    for conversation in conversations:
        user_id = conversation["user_id"]
        conv_id = conversation["id"]
        memory_start = agent.memory_file_size(user_id)

        main_thread = f"{conv_id}-main"
        for turn in conversation.get("turns", []):
            agent.reply(user_id, main_thread, turn)
            thread_agent_tokens[main_thread] = agent.token_usage(main_thread)
            thread_prompt_tokens[main_thread] = agent.prompt_token_usage(main_thread)
            max_compactions = max(max_compactions, agent.compaction_count(main_thread))

        for index, question in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{conv_id}-recall-{index}"
            response = agent.reply(user_id, recall_thread, question["question"])
            answer = response.get("answer", "")
            expected = question.get("expected_contains", [])

            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            recall_count += 1

            thread_agent_tokens[recall_thread] = agent.token_usage(recall_thread)
            thread_prompt_tokens[recall_thread] = agent.prompt_token_usage(recall_thread)

        memory_end = agent.memory_file_size(user_id)
        total_memory_growth += max(0, memory_end - memory_start)

    if recall_count == 0:
        recall_count = 1

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(thread_agent_tokens.values()),
        prompt_tokens_processed=sum(thread_prompt_tokens.values()),
        recall_score=total_recall / recall_count,
        response_quality=total_quality / recall_count,
        memory_growth_bytes=total_memory_growth,
        compactions=max_compactions,
    )


def _sum_agent_tokens(agent, thread_id: str, current: int) -> int:
    usage = agent.token_usage(thread_id)
    return current + usage


def _sum_prompt_tokens(agent, thread_id: str, current: int) -> int:
    usage = agent.prompt_token_usage(thread_id)
    return current + usage


def format_rows(rows: list[BenchmarkRow]) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_rows = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]

    if tabulate:
        return tabulate(table_rows, headers=headers, tablefmt="github")

    lines = [" | ".join(headers)]
    lines.append(" | ".join(["---"] * len(headers)))
    for table_row in table_rows:
        lines.append(" | ".join(str(value) for value in table_row))
    return "\n".join(lines)


def _run_suite(title: str, dataset_path: Path, config) -> str:
    conversations = load_conversations(dataset_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config),
        run_agent_benchmark("Advanced", advanced, conversations, config),
    ]
    return f"## {title}\n\n{format_rows(rows)}\n"


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("Day 17 Benchmark - Memory Systems for AI Agent\n")
    print(_run_suite("Standard Benchmark", standard_path, config))
    print(_run_suite("Long-Context Stress Benchmark", stress_path, config))


if __name__ == "__main__":
    main()
