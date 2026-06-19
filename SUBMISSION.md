# Hướng dẫn nộp bài — Day 17 Memory Systems

## Checklist hoàn thành

- [x] `config.py` — load `.env`, multi-provider
- [x] `model_provider.py` — OpenAI, Gemini, Anthropic, Ollama, OpenRouter, custom/Antco
- [x] `memory_store.py` — User.md, compact memory, confidence threshold, noise filter
- [x] `agent_baseline.py` — short-term only
- [x] `agent_advanced.py` — User.md + compact + LangGraph tools
- [x] `benchmark.py` — 6 cột chỉ số theo rubric
- [x] `test_agents.py` — 8 tests (profile, compact, recall, guardrail)
- [x] `ANALYSIS.md` — phân tích trade-off + bonus
- [x] Demo web — `demo_server.py` + `static/`

## Chuẩn bị trước khi nộp

1. Copy `.env.example` → `.env` và điền API key (không commit `.env`).
2. Cài dependencies:

```powershell
cd d:\solution\Day17-Track03-MemorySystems4Agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Chạy test:

```powershell
cd src
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest test_agents.py -v
```

(Nếu máy cài `deepeval` global, cần tắt autoload plugin để tránh xung đột OpenTelemetry.)

Kỳ vọng: **8 passed**.

4. Chạy benchmark:

```powershell
python benchmark.py
```

Kỳ vọng: Advanced recall > Baseline; stress test Advanced prompt tokens thấp hơn + compactions > 0.

5. (Tùy chọn) Demo live:

```powershell
python demo_server.py
```

Mở http://127.0.0.1:8765 — chat song song Baseline vs Advanced, đổi Thread ID để test cross-session.

6. (Tùy chọn) Kiểm tra gateway Antco:

```powershell
python test_proxy.py
```

## File cần nộp

| File / thư mục | Mô tả |
|---|---|
| `src/` | Toàn bộ implementation |
| `data/` | Benchmark input |
| `ANALYSIS.md` | Phân tích kết quả |
| `requirements.txt` | Dependencies |
| `.env.example` | Template cấu hình (không có secret) |
| `README.md` | Hướng dẫn chạy |

**Không nộp:** `.env`, `.venv/`, `state/`, `__pycache__/`

## Rubric mapping

| Mức điểm | Yêu cầu | Trạng thái |
|---|---|---|
| 0–60 | Baseline + Advanced cơ bản | Done |
| 60–75 | Benchmark + pytest | Done |
| 75–90 | ANALYSIS.md trade-offs | Done |
| 90–100 | Bonus guardrail | Done (confidence, correction, noise) |

## Ghi chú demo / screenshot (nếu giảng viên yêu cầu)

1. Screenshot UI demo: hai cột Baseline vs Advanced cùng câu hỏi recall.
2. Screenshot đổi Thread ID — Advanced nhớ tên/nghề, Baseline không.
3. Screenshot output `python benchmark.py` (bảng standard + stress).
4. Screenshot `User.md` trong panel Advanced sau vài lượt chat.

## Biến môi trường quan trọng

```env
LLM_PROVIDER=antco
LLM_MODEL=gemini-3-flash
CUSTOM_BASE_URL=https://ai-gateway.antco.ai
CUSTOM_API_KEY=<your-key>
FORCE_OFFLINE=false
PROFILE_CONFIDENCE_THRESHOLD=0.7
COMPACT_THRESHOLD_TOKENS=1200
COMPACT_KEEP_MESSAGES=6
```

`FORCE_OFFLINE=true` — chạy benchmark/test không cần API (đủ cho chấm điểm cơ bản).
