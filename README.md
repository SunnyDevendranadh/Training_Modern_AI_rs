# Training Modern AI

[![CI](https://github.com/SunnyDevendranadh/Training_Modern_AI_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SunnyDevendranadh/Training_Modern_AI_rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange)](https://www.rust-lang.org)

**Interactive economics of frontier transformer inference, end-to-end in Rust.**

A learning module that lets you explore — with real numbers, on your own
hardware assumptions — questions that come up every time someone tries to
reason about modern AI infrastructure:

- *Why does inference latency floor at ~20 ms regardless of hardware?*
- *Why is the batch sweet spot ≈ 2,000 sequences for sparse MoE models?*
- *Why is the context-length wall memory bandwidth, not FLOPs?*
- *Why do frontier labs over-train ~100× past Chinchilla optimal?*
- *Why does Recursive Planner+Worker scale near-linearly when other
  multi-agent strategies collapse?*

Every page is a parameter form + computed stats + an SVG chart + a results
table. Change a number, see the answer. URLs are shareable.

## Quick start

```bash
git clone https://github.com/SunnyDevendranadh/Training_Modern_AI_rs.git
cd Training_Modern_AI_rs
cargo run -p web
# → open http://127.0.0.1:2718
```

Requires Rust 1.75+ ([install via rustup](https://rustup.rs)).

That's it. No database, no Docker, no Node, no `npm install`. The entire app
is one Rust binary that serves HTML and SVG.

## What's inside

The app has nine pages, each one parameter-driven:

| Page | What it teaches |
|------|-----------------|
| `/roofline` | Total latency = max(compute, weights + KV cache). Find the balance batch where compute time meets memory time. |
| `/cost` | Cost per million tokens. The compute floor is asymptotic; the *knee* is the practical operating point. |
| `/context` | Why long context is a memory-bandwidth wall. Compute the exact context where KV reads catch the weight fetch. |
| `/scaling` | Pretrain : inference token ratio. There's no single optimum — the page explains what each ratio means in over-training terms. |
| `/pricing` | OpenAI sandbox-hour vs. Anthropic session-hour pricing. Find the break-even and watch model cost dominate at scale. |
| `/agents-md` | Vercel's empirical finding: passive AGENTS.md context peaks at ~8 KB; active retrieval caps at ~79%. |
| `/coordination` | Cursor's four multi-agent strategies. Watch Equal collapse past N=20; Recursive scale near-linearly. |
| `/throughput` | Blocking review vs. minimally-blocking merge. Where the speedup comes from when agent PR output exceeds human capacity. |

A CLI is included too — run all 14 underlying experiments as text summaries:

```bash
cargo run -p cli                  # run all experiments
cargo run -p cli -- run roofline  # one experiment
```

## Architecture

```
crates/
├── physics/      Pure-function library: 14 modules, 71 unit tests.
│                 Roofline, cost, KV cache, MoE, scaling, agents,
│                 reviews, knowledge entropy, throughput, pricing.
│                 No I/O, no state, no allocation in hot paths.
├── experiments/  14 self-contained text experiments built on `physics`.
│                 14 unit tests.
├── web/          Axum server with server-rendered HTML.
│                 ~40 lines of vanilla JS (no external deps) for
│                 live form updates without page reloads.
└── cli/          Single-binary entry point for both experiments and
                  the web server.
```

The web app is server-rendered: each route runs the physics in Rust and
returns a complete HTML document. A small embedded JavaScript snippet
hooks `change` and `input` events on the parameter form and refetches
just the results section in place. **No external CDN. No tracking. No
analytics. Runs entirely on localhost.**

## Examples

Once the server is running, copy these URLs into your browser:

```
# Roofline at FP4 — weight fetch halves to 70 ms
http://127.0.0.1:2718/roofline?bytes_per_param=0.5

# Context wall at small batch — KV catches weights at ~34M tokens
http://127.0.0.1:2718/context?batch=10

# Scaling at frontier ratio — ~130× over-training
http://127.0.0.1:2718/scaling?ratio=1

# Pricing at high model cost — break-even shifts
http://127.0.0.1:2718/pricing?model_per_million=10
```

## Testing

```bash
cargo test --workspace      # all 85 tests
cargo fmt --all -- --check  # formatting
cargo clippy --workspace -- -D warnings   # lints
```

End-to-end smoke test of every web route — see
[`crates/web/tests/`](crates/web/tests/).

## Contributing

Contributions welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for development
setup, code style, and the PR process.

Found a security issue? Please follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## Acknowledgements

This is a Rust rewrite of the
[original Python/Marimo project](https://github.com/SunnyDevendranadh/Training_Modern_AI),
which itself draws on:
- Roofline analysis and inference economics (DeepSeek-V3, Blackwell-NVL72)
- Cursor's multi-agent coordination experiments (1 → 100 agents)
- Vercel's AGENTS.md effectiveness findings
- Chinchilla scaling laws and over-training observations from frontier labs

## License

MIT — see [LICENSE](LICENSE).
