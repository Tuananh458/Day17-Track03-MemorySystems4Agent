# Student Scaffold

This `src/` folder contains the completed Day 17 lab implementation.

- Baseline and Advanced agents with offline + LangGraph live paths
- LangGraph uses `create_agent`, `MemorySaver`, tools, and `SummarizationMiddleware`
- Benchmark: standard + long-context stress
- Demo UI: `python demo_server.py`

Suggested flow:

1. `config.py`
2. `memory_store.py`
3. `agent_baseline.py` + `agent_advanced.py`
4. `langgraph_runtime.py`
5. `benchmark.py`
6. `test_agents.py`
7. `python demo_server.py` for interactive demo

Datasets are available at the repo root in `data/`.
