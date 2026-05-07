# Training Modern AI — Rust Edition

End-to-end learning module for transformer inference, GPU economics, and agent harness engineering. **Complete Rust rewrite** of the original [Python/Marimo project](https://github.com/SunnyDevendranadh/Training_Modern_AI).

## Quick start

```bash
# Run all 85 tests
cargo test

# Run all 14 experiments (text summaries)
cargo run -p cli

# Start web server
cargo run -p web
# → http://127.0.0.1:2718

# Run a single experiment
cargo run -p cli -- run roofline
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Cargo Workspace                    │
├─────────┬──────────────┬──────────────┬─────────────┤
│ physics │ experiments  │     web      │     cli     │
│ (core)  │ (14 static)  │ (Axum SSR)   │ (entry)      │
│ 71 tests│  14 tests    │ 9 HTML pages │              │
├─────────┴──────────────┴──────────────┴─────────────┤
│                85 total tests passing                 │
└─────────────────────────────────────────────────────┘
```

## Modules

### 🛠️ ML Infrastructure (physics crate + experiments)

| # | Topic | Core equation |
|---|-------|---------------|
| 1 | Roofline | `t_total ≥ max(t_compute, t_mem_weights + t_mem_kv)` |
| 2 | Batch & Latency | Cost floor = compute/token, Latency floor ~20ms |
| 3 | KV Cache | `t_mem_kv = B·ctx·bpt / BW` |
| 4 | MoE Routing | 256 experts, 37B active / 700B total |
| 5 | Pipeline | Bubble = `(S-1)/(S-1+M)` |
| 6 | Memory Tiers | HBM → DDR → Flash → Rematerialize |
| 7 | Scaling Laws | ~100× over-training vs Chinchilla |

### 🎛️ Agent Harness (physics crate + experiments)

| # | Topic | Core claim |
|---|-------|------------|
| 8 | Harness Effectiveness | 8KB AGENTS.md = 100% success (Vercel) |
| 9 | Throughput vs Perfection | Waiting > correcting at scale |
| 10 | Multi-Agent Coordination | Recursive Planner+Worker scales |
| 11 | Knowledge Decay | Continuous GC beats weekly cleanup |
| 12 | Agent Reviews | Self→Cross→Human pipeline |
| 13 | Harness Pricing | OpenAI sandbox vs Anthropic MCP |
| 14 | Context Economics | 8KB uses <1% of context window |

## Web app

Every page is **server-rendered HTML**. Forms submit via GET, the server runs
the physics, and the response is a complete document with embedded SVG charts.
No client JavaScript is required — the browser only renders.

```
GET /             → home, key insights, module index
GET /roofline     → latency vs batch size (form + chart + table)
GET /cost         → $ per million tokens (form + chart + table)
GET /context      → KV cache wall (form + chart + table)
GET /scaling      → pretrain vs inference cost (form + chart + table)
GET /pricing      → OpenAI vs Anthropic (form + chart + table)
GET /agents-md    → AGENTS.md effectiveness (form + chart + table)
GET /coordination → multi-agent scaling (form + chart + table)
GET /throughput   → blocking vs minimally-blocking (form + chart + table)
GET /api/health   → {"status": "ok"}
```

URLs are shareable: every parameter is a query string. Example:

```
http://127.0.0.1:2718/roofline?n_active=37e9&context=131072&bytes_per_param=0.5
```

## Differences from Python original

| Aspect | Python (Marimo) | Rust |
|--------|-----------------|------|
| UI | Marimo reactive notebook | Server-rendered HTML + SVG |
| Client deps | Plotly + JS | None — plain HTML, no JS |
| Charts | Plotly (interactive) | Pure-Rust SVG (zero deps) |
| Package mgmt | uv | Cargo |
| Types | Dynamic | Static, serde-serializable |
| Concurrency | Single-threaded | Tokio async |
| Tests | None (notebook) | 85 unit/integration tests |

## License

MIT — see original project
