# AGENTS.md — Training Modern AI (Rust Edition)

Single source of truth for AI agents working on this repo.

---

## 1. Project overview

**Name:** Training Modern AI — Rust Edition
**Stack:** Rust (2021 edition). **Cargo** for packaging. **Axum** for the web server.
**Original:** Python/Marimo interactive learning module, fully rewritten in Rust.

Two modules:
- **ML Infrastructure & Transformer Inference** — roofline analysis, batch economics, KV cache, MoE, parallelism, memory hierarchy, scaling laws
- **Agent Harness Engineering** — AGENTS.md pattern, knowledge versioning, agent reviews, throughput vs perfection, multi-agent coordination, pricing models

## 2. Workspace structure

```
training-modern-ai-rs/
├── Cargo.toml                 # workspace root
├── README.md
├── AGENTS.md                  # this file
├── crates/
│   ├── physics/               # core computation library (14 modules)
│   │   ├── src/
│   │   │   ├── lib.rs         # re-exports
│   │   │   ├── constants.rs   # FLOPS, MEM_BW, N_TOTAL, etc.
│   │   │   ├── types.rs       # LatencyParams, CoordinationStrategy, etc.
│   │   │   ├── latency.rs     # t_compute, t_mem_weights, t_mem_kv, total_latency
│   │   │   ├── cost.rs        # cost_per_million_tokens, rack_cost_per_sec
│   │   │   ├── moe.rs         # MoE routing simulation
│   │   │   ├── pipeline.rs    # bubble ratio, throughput
│   │   │   ├── agents.rs      # multi-agent coordination (Cursor's 4 iterations)
│   │   │   ├── knowledge.rs   # entropy simulation, AGENTS.md effectiveness
│   │   │   ├── reviews.rs     # review pipeline simulation
│   │   │   ├── pricing.rs     # OpenAI vs Anthropic pricing
│   │   │   ├── scaling.rs     # scaling laws, over-training
│   │   │   └── throughput.rs  # merge strategy trade-offs
│   │   └── tests/             # (inline module tests — 71 tests)
│   │
│   ├── experiments/           # 14 static experiments
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── plot_utils.rs  # shared utilities
│   │   │   ├── ml_infra/      # 7 ML infra experiments
│   │   │   └── harness/       # 7 harness experiments
│   │   └── tests/             # (inline — 14 tests)
│   │
│   ├── web/                   # Axum web server (Marimo replacement)
│   │   ├── src/
│   │   │   ├── main.rs        # binary entry point
│   │   │   ├── lib.rs
│   │   │   ├── routes.rs      # 6 JSON API endpoints
│   │   │   └── templates/
│   │   │       └── index.html
│   │   └── tests/
│   │
│   └── cli/                   # CLI (run experiments, serve web)
│       └── src/
│           └── main.rs
└── assets/                    # generated output
```

## 3. Quick start

```bash
cargo test                          # run all 85 tests
cargo run -p cli                    # run all 14 experiments (text summary)
cargo run -p cli -- run roofline    # run a single experiment
cargo run -p web                    # start web server (http://127.0.0.1:2718)
PORT=8080 cargo run -p web          # custom port
```

## 4. Conventions

- **Pure functions**: All physics/agent computations are pure, side-effect-free functions. No global state.
- **Tests inline**: Tests live in `#[cfg(test)] mod tests` within each source file. Run with `cargo test -p physics`.
- **ndarray**: Use `ndarray::Array1<f64>` for vectorized computations. Avoid raw loops where ndarray operations suffice.
- **Constants**: All hardware/model constants live in `physics::constants`. Use them, don't re-derive.
- **Types**: Shared types in `physics::types`. Serde-serializable for API responses.
- **Web API**: JSON-only. Each endpoint returns computed results. The HTML template is static with JS fetch.
- **No external plotting**: The experiments crate outputs text summaries. Full chart generation remains in Python `experiments/` and `Harness/experiments/` directories.

## 5. Crate dependency graph

```
cli ──────────┐
              ├── experiments ── physics
web ──────────┘
```

- `physics` has no external deps beyond ndarray, serde, rand
- `experiments` depends on `physics`
- `cli` depends on `experiments` + clap
- `web` depends on `physics` + axum + tokio

## 6. Test count

| Crate | Tests |
|-------|-------|
| physics | 71 |
| experiments | 14 |
| web | 0 (integration) |
| cli | 0 (integration) |
| **Total** | **85** |

## 7. Key insights (from original Python project)

1. **Latency floor:** ~20 ms — physics, not engineering
2. **Batch sweet spot:** ~2,000 sequences for modern sparse models
3. **Rack boundary:** one NVLink domain bounds MoE expert parallelism
4. **Context wall:** memory bandwidth, not compute
5. **Over-training:** ~100× vs Chinchilla — driven by inference demand
6. **Agent inference cost:** 10–50× a single chat turn
7. **Harness > code:** human leverage shifts to environment design
8. **AGENTS.md ≈ 8 KB:** Vercel's empirical sweet spot
9. **Recursive Planner+Worker:** the only multi-agent pattern that scales
