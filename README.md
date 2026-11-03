# TURFVISION (VAR) — 8v8 Turf AI Analytics Platform

A monocular (single 4K 60fps ultra-wide smartphone) 8v8 football match analytics and 3D goal reconstruction platform.

## Verification-Driven Agentic Architecture (8 Foundational Documents)

This project is governed by strict deterministic agent execution rules:

1. [AGENT_RULES.md](AGENT_RULES.md) — Anti-skimming protocol and execution laws.
2. [ARCHITECTURE_AND_DESIGN.md](ARCHITECTURE_AND_DESIGN.md) — System blueprint, pitch geometry, and kinematics math.
3. [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — Strict inter-service JSON schemas (Pydantic & Zod).
4. [GOLD_STANDARDS.md](GOLD_STANDARDS.md) — Deterministic quantitative acceptance criteria.
5. [MODEL_AND_HARDWARE_MANIFEST.md](MODEL_AND_HARDWARE_MANIFEST.md) — Exact weights, HuggingFace/ONNX checkpoints, and sequential VRAM offload rules.
6. [EDGE_CASES_AND_GUARDRAILS.md](EDGE_CASES_AND_GUARDRAILS.md) — 8v8 turf cage physics, touchline filtering, GK logic, and coordinate systems.
7. [PHASED_TASKS.md](PHASED_TASKS.md) — Atomic task execution backlog with pass/fail verification gates.
8. [PROGRESS_LOG.md](PROGRESS_LOG.md) — Agent execution ledger and stdout metrics history.

All documents are also archived under [`.agent/`](.agent/).
