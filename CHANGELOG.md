# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Server-side partial updates via vanilla JS (replaces external HTMX CDN).
  The ~40-line inline script eliminates the third-party CDN dependency while
  preserving live form updates and graceful no-JS fallback.
- Open-source housekeeping: LICENSE (MIT), CONTRIBUTING.md, SECURITY.md,
  issue templates, PR template, Dependabot config, and upgraded CI pipeline
  (fmt + clippy + audit + multi-OS matrix).
- Workspace-level `rust-version`, `homepage`, `readme`, `keywords`, and
  `categories` fields propagated to all crates for crates.io readiness.
- Fixed Cargo.toml `repository` URL (was pointing at the Python predecessor).

### Changed
- CI workflow renamed from `rust.yml` → `ci.yml`; now runs `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test`, and `cargo audit` on Ubuntu and
  macOS.
- LICENSE copyright updated to include the project owner by name.

## [0.1.0] — 2025-05-07

### Added
- Complete Rust workspace with four crates: `physics`, `experiments`, `web`,
  `cli`.
- `physics` crate: 14 pure-function modules (roofline, cost, KV cache, MoE,
  pipeline, memory tiers, scaling laws, agents, knowledge entropy, reviews,
  pricing, throughput) — 71 unit tests.
- `experiments` crate: 14 static text-mode experiments — 14 unit tests.
- `web` crate: Axum server with 8 interactive parameter pages + health
  endpoint; server-rendered HTML + pure-Rust SVG charts; no client JS or
  external dependencies.
- `cli` crate: single-binary entry point for experiments and web server.
- GitHub Actions CI (build + test on Ubuntu).
- AGENTS.md: AI agent harness documentation.

[Unreleased]: https://github.com/SunnyDevendranadh/Training_Modern_AI_rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SunnyDevendranadh/Training_Modern_AI_rs/releases/tag/v0.1.0
