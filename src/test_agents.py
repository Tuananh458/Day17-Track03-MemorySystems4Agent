from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_candidates,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    provider = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.2,
        api_key="test-key",
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        profile_confidence_threshold=0.7,
        profile_decay_half_life_days=7.0,
        profile_decay_min_confidence=0.5,
        model=provider,
        judge_model=provider,
        force_offline=True,
    )

def test_langgraph_agents_build_when_dependencies_exist() -> None:
    from langgraph_runtime import langgraph_available

    if not langgraph_available():
        return

    config = make_config(Path("."))
    baseline = BaselineAgent(config, force_offline=False)
    advanced = AdvancedAgent(config, force_offline=False)
    assert baseline.langgraph_agent is None or baseline.checkpointer is not None
    assert advanced.langgraph_agent is None or advanced.checkpointer is not None


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    store.write_text("user1", "# User Profile\n\n- name: A\n")
    assert "name: A" in store.read_text("user1")

    store.upsert_fact("user1", "location", "Huế")
    assert "location: Huế" in store.read_text("user1")

    changed = store.edit_text("user1", "Huế", "Đà Nẵng")
    assert changed is True
    assert "location: Đà Nẵng" in store.read_text("user1")
    assert store.file_size("user1") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "long-thread"

    for index in range(12):
        manager.append(
            thread_id,
            "user",
            f"Tin nhắn số {index} " + ("x" * 80),
        )

    assert manager.compaction_count(thread_id) >= 1
    context = manager.context(thread_id)
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    advanced = AdvancedAgent(config, force_offline=True)
    advanced.reply("u1", "thread-a", "Chào bạn, mình tên là DũngCT.")
    advanced.reply("u1", "thread-a", "Mình ở Huế và làm MLOps engineer.")
    recall = advanced.reply("u1", "thread-b", "Mình tên gì và làm nghề gì?")

    assert "DũngCT" in recall["answer"]
    assert "MLOps engineer" in recall["answer"]

    baseline = BaselineAgent(config, force_offline=True)
    baseline.reply("u1", "thread-a", "Chào bạn, mình tên là DũngCT.")
    baseline_recall = baseline.reply("u1", "thread-b", "Mình tên gì?")
    assert "DũngCT" not in baseline_recall["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    thread_id = "stress-thread"
    for index in range(20):
        message = f"Lượt {index}: " + ("benchmark memory systems " * 20)
        baseline.reply("u1", thread_id, message)
        advanced.reply("u1", thread_id, message)

    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
    assert advanced.compaction_count(thread_id) >= 1


def test_question_not_saved_to_profile() -> None:
    updates = extract_profile_updates("Mình tên gì và đang ở đâu?")
    assert updates == {}


def test_correction_overwrites_location(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    agent.reply("u1", "t1", "Mình tên là DũngCT, mình ở Đà Nẵng.")
    agent.reply("u1", "t2", "Đính chính: giờ mình đang ở Huế, không còn ở Đà Nẵng nữa.")
    recall = agent.reply("u1", "t3", "Hiện tại mình đang ở đâu?")

    assert "Huế" in recall["answer"]
    profile = agent.get_profile_text("u1")
    assert "Huế" in profile
    assert "Đà Nẵng" not in profile


def test_noise_profession_not_saved() -> None:
    updates = extract_profile_updates(
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager cho đỡ phải ngồi canh pipeline, nhưng đó chỉ là câu đùa."
    )
    assert "profession" not in updates


def test_entity_extraction_has_structured_types() -> None:
    facts = extract_profile_candidates("Mình tên là DũngCT, mình ở Huế và làm MLOps engineer.")
    by_key = {fact.key: fact for fact in facts}
    assert by_key["name"].entity_type == "person"
    assert by_key["location"].entity_type == "location"
    assert by_key["profession"].entity_type == "profession"


def test_memory_decay_excludes_stale_fact(tmp_path: Path) -> None:
    import time

    store = UserProfileStore(
        tmp_path / "profiles",
        decay_half_life_days=7.0,
        min_effective_confidence=0.5,
    )
    store.upsert_entity("u1", "location", "Huế", confidence=0.85, entity_type="location")
    entities = store.load_entities("u1")
    entities["location"].updated_at = time.time() - (60 * 86_400)
    store.save_entities("u1", entities)

    assert "location" not in store.active_facts("u1")
    assert "Huế" not in store.get_active_profile_text("u1")


def test_remention_restores_decayed_fact(tmp_path: Path) -> None:
    import time

    store = UserProfileStore(
        tmp_path / "profiles",
        decay_half_life_days=7.0,
        min_effective_confidence=0.5,
    )
    store.upsert_entity("u1", "favorite_drink", "cà phê sữa đá", confidence=0.9, entity_type="preference")
    entities = store.load_entities("u1")
    entities["favorite_drink"].updated_at = time.time() - (45 * 86_400)
    store.save_entities("u1", entities)
    assert "favorite_drink" not in store.active_facts("u1")

    store.upsert_entity("u1", "favorite_drink", "cà phê sữa đá", confidence=0.9, entity_type="preference")
    assert "favorite_drink" in store.active_facts("u1")
    assert store.load_entities("u1")["favorite_drink"].mention_count >= 2
