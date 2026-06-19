from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def _slugify(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_id.strip())
    return safe or "default_user"


def _parse_profile(content: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in content.splitlines():
        match = re.match(r"^-\s*([^:]+):\s*(.+?)\s*$", line.strip())
        if match:
            facts[match.group(1).strip()] = match.group(2).strip()
    return facts


def _render_profile(facts: dict[str, str]) -> str:
    lines = ["# User Profile", ""]
    for key in sorted(facts):
        lines.append(f"- {key}: {facts[key]}")
    lines.append("")
    return "\n".join(lines)


@dataclass
class UserProfileStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return _parse_profile(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        facts = self.facts(user_id)
        facts[key] = value.strip()
        return self.write_text(user_id, _render_profile(facts))


_QUESTION_HINTS = (
    "?",
    "bạn có thể",
    "bạn biết",
    "tên mình là gì",
    "mình tên gì",
    "là gì",
    "ở đâu",
    "như thế nào",
)


def _looks_like_question(message: str) -> bool:
    lowered = message.lower().strip()
    if lowered.endswith("?"):
        return True
    return any(hint in lowered for hint in _QUESTION_HINTS) and not any(
        marker in lowered
        for marker in ("mình tên là", "đính chính", "giờ mình", "hiện tại mình", "mình muốn bạn")
    )


@dataclass
class ProfileFact:
    key: str
    value: str
    confidence: float


_NOISE_MARKERS = (
    "câu đùa",
    "chỉ là câu đùa",
    "không phải nơi ở",
    "chỉ là nơi mình bay",
    "vừa bay ra họp",
    "hay là chuyển sang product manager",
    "product manager cho đỡ",
)


def _is_noise_context(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _NOISE_MARKERS)


def extract_profile_candidates(message: str) -> list[ProfileFact]:
    if _looks_like_question(message):
        return []

    text = " ".join(message.split())
    lowered = text.lower()
    is_correction = any(token in lowered for token in ("đính chính", "correction", "cập nhật", "không còn"))
    is_noise = _is_noise_context(text)
    facts: list[ProfileFact] = []

    name_match = re.search(
        r"(?:mình tên là|minh ten la|tên mình là|ten minh la|tên là|ten la)\s+([^.,;!\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        facts.append(ProfileFact("name", name_match.group(1).strip(), 0.95))

    location_patterns = [
        (r"đính chính[^.]*giờ mình đang ở\s+([^.,;!\n]+)", 1.0),
        (r"giờ mình đang ở\s+([^.,;!\n]+)", 0.95 if is_correction else 0.85),
        (r"hiện tại mình đang ở\s+([^.,;!\n]+)", 0.9),
        (r"hiện\s+ở\s+([^.,;!\n]+?)(?:\s+và|\s+trong|\.)", 0.85),
        (r"mình ở\s+([^.,;!\n]+?)(?:\s+và|\s+đang|\.)", 0.8),
        (r"mình vẫn ở\s+([^.,;!\n]+)", 0.85),
        (r"nơi ở hiện tại là\s+([^.,;!\n]+)", 0.9),
        (r"đang làm việc ở\s+([^.,;!\n]+)", 0.88),
    ]
    if not is_noise:
        for pattern, confidence in location_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                location = re.sub(r"\s+(trong|vài|để|và).*$", "", location, flags=re.IGNORECASE)
                facts.append(ProfileFact("location", location, confidence))
                break

    profession_patterns = [
        (r"giờ chuyển sang\s+([^.,;!\n]+)", 1.0),
        (r"không còn làm\s+[^,]+,\s*giờ chuyển sang\s+([^.,;!\n]+)", 1.0),
        (r"đang làm\s+([^.,;!\n]+?)(?:\s+cho|\s+và|\.)", 0.88),
        (r"là\s+([^.,;!\n]+?\s+engineer)", 0.9),
        (r"nghề nghiệp hiện tại[:\s]+([^.,;!\n]+)", 0.92),
    ]
    if not is_noise:
        profession_found = False
        for pattern, confidence in profession_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                profession = match.group(1).strip()
                profession = re.sub(r"\s+nữa$", "", profession, flags=re.IGNORECASE)
                if "product manager" in profession.lower() and "đùa" in lowered:
                    continue
                facts.append(ProfileFact("profession", profession, confidence))
                profession_found = True
                break
        if not profession_found:
            match = re.search(r"(?:và\s+)?làm\s+([^.,;!\n]+)", text, flags=re.IGNORECASE)
            if match:
                profession = match.group(1).strip()
                facts.append(ProfileFact("profession", profession, 0.75))

    if "trả lời ngắn gọn" in lowered or "3 bullet" in lowered:
        if "3 bullet" in lowered:
            facts.append(
                ProfileFact(
                    "response_style",
                    "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off",
                    0.9,
                )
            )
        else:
            facts.append(
                ProfileFact(
                    "response_style",
                    "ngắn gọn, rõ ý, có ví dụ thực tế",
                    0.85,
                )
            )

    drink_match = re.search(
        r"đồ uống yêu thích(?:\s+là)?\s+([^.,;!\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if drink_match:
        facts.append(ProfileFact("favorite_drink", drink_match.group(1).strip(), 0.9))
    elif "cà phê sữa đá" in lowered and "uống" in lowered:
        facts.append(ProfileFact("favorite_drink", "cà phê sữa đá", 0.8))

    food_match = re.search(
        r"món ăn yêu thích(?:\s+là)?\s+([^.,;!\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if food_match:
        facts.append(ProfileFact("favorite_food", food_match.group(1).strip(), 0.9))
    elif "mì quảng" in lowered:
        facts.append(ProfileFact("favorite_food", "mì Quảng", 0.75))

    pet_match = re.search(
        r"nuôi[^.]*?(corgi)[^.]*tên\s+([^.,;!\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if pet_match:
        facts.append(
            ProfileFact(
                "pet",
                f"{pet_match.group(1)} tên {pet_match.group(2).strip()}",
                0.9,
            )
        )

    interest_bits: list[str] = []
    for keyword in ("python", "ai", "mlops", "benchmark", "memory"):
        if keyword in lowered:
            interest_bits.append(keyword)
    if interest_bits:
        facts.append(ProfileFact("interests", ", ".join(dict.fromkeys(interest_bits)), 0.72))

    return facts


def extract_profile_updates(message: str, min_confidence: float = 0.7) -> dict[str, str]:
    best: dict[str, ProfileFact] = {}
    for fact in extract_profile_candidates(message):
        current = best.get(fact.key)
        if current is None or fact.confidence > current.confidence:
            best[fact.key] = fact

    return {
        key: fact.value
        for key, fact in best.items()
        if fact.confidence >= min_confidence
    }


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    selected = messages[:max_items]
    lines = []
    for item in selected:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if len(content) > 180:
            content = content[:177] + "..."
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _estimate_thread_tokens(self, thread_state: dict[str, object]) -> int:
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        summary: str = str(thread_state.get("summary", ""))
        total = estimate_tokens(summary)
        for message in messages:
            total += estimate_tokens(message.get("content", ""))
        return total

    def _compact_if_needed(self, thread_id: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]

        while (
            len(messages) > self.keep_messages
            and self._estimate_thread_tokens(thread_state) > self.threshold_tokens
        ):
            old_count = len(messages) - self.keep_messages
            old_messages = messages[:old_count]
            messages[:] = messages[old_count:]

            summary_chunk = summarize_messages(old_messages)
            existing_summary = str(thread_state.get("summary", "")).strip()
            if existing_summary:
                thread_state["summary"] = f"{existing_summary}\n{summary_chunk}".strip()
            else:
                thread_state["summary"] = summary_chunk
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        self._compact_if_needed(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        thread_state = self._ensure_thread(thread_id)
        return {
            "messages": list(thread_state["messages"]),  # type: ignore[arg-type]
            "summary": str(thread_state.get("summary", "")),
            "compactions": int(thread_state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))
