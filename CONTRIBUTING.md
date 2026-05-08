# Contributing to Training Modern AI

Thanks for your interest in improving this project. Contributions of all
kinds are welcome — bug reports, documentation, new physics modules, web UX
improvements, and translations.

By contributing you agree that your work will be released under the
project's [MIT License](LICENSE).

## Getting started

```bash
git clone https://github.com/SunnyDevendranadh/Training_Modern_AI_rs.git
cd Training_Modern_AI_rs

# Build and test
cargo build --workspace
cargo test --workspace          # 85 tests should pass

# Run the web app
cargo run -p web                # → http://127.0.0.1:2718

# Run a single experiment via CLI
cargo run -p cli -- run roofline
```

Requires:
- Rust 1.75 or newer ([rustup](https://rustup.rs))
- A modern terminal and a browser

## Project layout

| Crate | Purpose | When to edit |
|-------|---------|--------------|
| `physics` | Pure functions for all computations | Adding a new formula or refining an existing one |
| `experiments` | Text-mode reproductions of each module | Adding a new experiment summary |
| `web` | Axum server + HTML/SVG rendering | Adding a new page, fixing UX, improving charts |
| `cli` | Workspace entry point | Adding a new command flag |

## Code style

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs both. PRs that don't format cleanly or have new lint warnings will
fail.

Conventions in this codebase:
- **Pure functions** in `physics`. No global state, no I/O. Everything
  takes its inputs as arguments and returns the result.
- **`ndarray::Array1`** for vectorized math, not raw `Vec<f64>`.
- **Constants live in `physics::constants`**. Don't re-derive `FLOPS` etc.
- **Tests inline** in `#[cfg(test)] mod tests` blocks within each source
  file. Use `approx::assert_relative_eq!` for floating-point comparisons.
- **No client-side dependencies** in the web crate. The ~40 lines of
  vanilla JavaScript embedded in `render.rs` are the only JS we ship.
- **No `unsafe`** without strong justification. The codebase is currently
  100% safe Rust and we'd like to keep it that way.

## Adding a new page to the web app

A typical page route lives in `crates/web/src/routes.rs` and follows this
shape:

```rust
async fn my_page(
    headers: HeaderMap,
    Query(p): Query<MyParams>,
) -> Html<String> {
    // 1. Read parameters with sensible defaults
    let foo = p.foo.unwrap_or(SOMETHING);

    // 2. Run the physics (everything in `physics::` is pure)
    let result = physics::my_module::compute(foo, ...);

    // 3. Build the three pieces
    let intro = r#"<h1>...</h1><p class="subtitle">...</p>"#;
    let form_section = section("Parameters", &form(...));
    let results = format!("{stats}{chart}{table}", ...);

    // 4. Hand off to respond() — handles HX-Request fragmenting
    Html(respond(is_htmx(&headers), "Title", "/my-page", intro, &form_section, &results))
}
```

Then add the route in `create_router()` and a nav entry in `render::NAV`.

## Verifying physics correctness

When adding or changing a route's stats, **hand-derive the expected value**
from the formula and compare. The repository has a verification pattern:

1. Add a unit test in the appropriate `physics::*` module covering the
   underlying function.
2. Add an integration test in `crates/web/tests/` that hits the route at
   default parameters and asserts the rendered stat values.

A bug we shipped early on (and caught later) was that `/scaling` reported a
fake "optimal ratio" that was always the lower bound of the search range.
The lesson: don't trust labels — verify numbers.

## Pull request process

1. Fork the repo and create a topic branch from `master`.
2. Make your change. Add or update tests.
3. Run the full check locally:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
4. Open a PR. Reference any related issues. Describe what changed and why.
5. CI must pass. A maintainer will review.

Small, focused PRs are reviewed faster than sprawling ones. If you're
planning a big change, open an issue first to discuss the design.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).
Include:
- What you ran (full command, including any URL parameters)
- What you expected to see
- What you actually saw
- The Rust version and OS

For physics-related bugs (a stat shows a number that doesn't match the
formula), include the hand-derivation that contradicts the displayed value.

## Suggesting features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
The most welcome contributions are:
- New pages built on existing `physics::` modules
- New physics modules backed by published research (cite the paper)
- UX improvements: better charts, accessibility, mobile layout
- Documentation, examples, tutorials

## Security issues

Please do **not** open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the disclosure process.

## License of contributions

By submitting a pull request, you agree that your contribution is released
under the [MIT License](LICENSE) and that you have the right to make that
contribution.
