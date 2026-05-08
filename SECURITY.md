# Security Policy

## Supported versions

Only the latest commit on the default branch (`master`) is actively maintained.
Older releases are not backported.

## Scope

This is a **local-only educational tool** that runs an HTTP server bound to
`127.0.0.1` by default. It accepts no user accounts, stores no data, and has
no internet-facing attack surface in its default configuration.

The security boundary that matters:

| Area | In scope |
|------|----------|
| Server-side HTML injection (XSS in rendered pages) | Yes |
| Supply-chain: dependency vulnerabilities | Yes |
| Denial of service against the local server | Low priority |
| Issues only triggered by setting `HOST=0.0.0.0` explicitly | Yes — note in report |
| Physics calculation accuracy (not a security issue) | No — use a bug report |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private vulnerability reporting instead:

1. Go to the [Security tab](https://github.com/SunnyDevendranadh/Training_Modern_AI_rs/security)
   of this repository.
2. Click **"Report a vulnerability"**.
3. Fill in the form: describe the issue, reproduction steps, and impact.

This keeps the report private until a fix is ready. We will acknowledge
receipt within **5 business days** and aim to ship a fix within **30 days**
for confirmed issues.

## After a fix ships

Once the vulnerability is patched, we will:

- Credit the reporter in the commit message and release notes (unless you
  prefer anonymity).
- Publish a GitHub Security Advisory with the CVE (if applicable).

## Dependency auditing

This repository uses [`cargo audit`](https://crates.io/crates/cargo-audit) in
CI to catch known-vulnerable dependency versions automatically.

To run locally:

```bash
cargo install cargo-audit
cargo audit
```

## Threat model note

Default deployment is `localhost:2718`. All computations are pure CPU math on
user-supplied numeric parameters. There is no database, no authentication, no
file system writes, no external network calls, and no client-side JS from
external origins. The attack surface is intentionally minimal.
