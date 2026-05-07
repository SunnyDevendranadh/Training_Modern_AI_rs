//! HTML page chrome: layout, navigation, form helpers.
//!
//! Every page is rendered server-side as a complete HTML document. No
//! client JavaScript is required — forms submit via GET and the server
//! returns a fully rendered page with computed results.

use std::fmt::Write;

use crate::chart::html_escape;

/// All navigation entries, used to render the top nav on every page.
pub const NAV: &[(&str, &str, &str)] = &[
    ("/", "Home", "Overview & key insights"),
    ("/roofline", "Roofline", "Latency vs batch size"),
    ("/cost", "Cost", "Cost per million tokens"),
    ("/context", "Context", "KV cache wall"),
    ("/scaling", "Scaling", "Pre-train vs inference"),
    ("/pricing", "Pricing", "OpenAI vs Anthropic"),
    ("/agents-md", "AGENTS.md", "Passive context sweet spot"),
    ("/coordination", "Coordination", "Multi-agent strategies"),
    ("/throughput", "Throughput", "Blocking vs minimally-blocking"),
];

pub fn page(title: &str, current_path: &str, body: &str) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html>\n<html lang=\"en\"><head>");
    let _ = write!(
        s,
        "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{t} — Training Modern AI</title>",
        t = html_escape(title)
    );
    s.push_str(STYLE);
    s.push_str("</head><body>");

    // Header
    s.push_str(r#"<header class="hdr"><div class="hdr-inner"><a class="logo" href="/">Training Modern AI <span class="muted">— Rust</span></a><nav class="nav">"#);
    for (path, label, _) in NAV {
        let active = if *path == current_path { " active" } else { "" };
        let _ = write!(
            s,
            r#"<a class="nav-link{active}" href="{p}">{l}</a>"#,
            p = path,
            l = html_escape(label),
        );
    }
    s.push_str("</nav></div></header>");

    // Body
    let _ = write!(s, r#"<main class="container">{}</main>"#, body);

    s.push_str(r#"<footer class="ftr">85 tests · 14 experiments · pure Rust · server-rendered</footer>"#);
    s.push_str("</body></html>");
    s
}

/// A computed metric to highlight at the top of a results section.
pub struct Stat {
    pub label: String,
    pub value: String,
    pub hint: String,
}

impl Stat {
    pub fn new(label: impl Into<String>, value: impl Into<String>, hint: impl Into<String>) -> Self {
        Self { label: label.into(), value: value.into(), hint: hint.into() }
    }
}

pub fn stats_grid(stats: &[Stat]) -> String {
    let mut s = String::from(r#"<div class="stats">"#);
    for st in stats {
        let _ = write!(
            s,
            r#"<div class="stat"><div class="stat-label">{l}</div><div class="stat-value">{v}</div><div class="stat-hint">{h}</div></div>"#,
            l = html_escape(&st.label),
            v = html_escape(&st.value),
            h = html_escape(&st.hint),
        );
    }
    s.push_str("</div>");
    s
}

pub struct Field<'a> {
    pub name: &'a str,
    pub label: &'a str,
    pub hint: &'a str,
    pub value: String,
    pub kind: FieldKind<'a>,
}

pub enum FieldKind<'a> {
    Number {
        step: &'a str,
        min: Option<&'a str>,
    },
    Select {
        options: &'a [(&'a str, &'a str)],
    },
}

pub fn form(action: &str, fields: &[Field]) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        r#"<form class="params" method="get" action="{a}"><div class="params-grid">"#,
        a = action,
    );
    for f in fields {
        let _ = write!(
            s,
            r#"<label class="field"><span class="field-label">{l}</span>"#,
            l = html_escape(f.label),
        );
        match &f.kind {
            FieldKind::Number { step, min } => {
                let min_attr = min
                    .map(|m| format!(r#" min="{}""#, m))
                    .unwrap_or_default();
                let _ = write!(
                    s,
                    r#"<input type="number" inputmode="decimal" name="{n}" value="{v}" step="{step}"{min_attr}>"#,
                    n = f.name,
                    v = html_escape(&f.value),
                );
            }
            FieldKind::Select { options } => {
                let _ = write!(s, r#"<select name="{n}">"#, n = f.name);
                for (val, label) in *options {
                    let sel = if *val == f.value { " selected" } else { "" };
                    let _ = write!(
                        s,
                        r#"<option value="{v}"{sel}>{l}</option>"#,
                        v = html_escape(val),
                        l = html_escape(label),
                    );
                }
                s.push_str("</select>");
            }
        }
        let _ = write!(
            s,
            r#"<span class="field-hint">{h}</span></label>"#,
            h = html_escape(f.hint),
        );
    }
    s.push_str(r#"</div><div class="params-actions"><button type="submit">Recompute</button><button type="reset" formaction formnovalidate onclick="this.form.reset();">Reset form</button></div></form>"#);
    s
}

/// Render a simple two-column table.
pub fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = String::from(r#"<div class="table-wrap"><table class="data"><thead><tr>"#);
    for h in headers {
        let _ = write!(s, r#"<th>{}</th>"#, html_escape(h));
    }
    s.push_str("</tr></thead><tbody>");
    for row in rows {
        s.push_str("<tr>");
        for cell in row {
            let _ = write!(s, r#"<td>{}</td>"#, html_escape(cell));
        }
        s.push_str("</tr>");
    }
    s.push_str("</tbody></table></div>");
    s
}

pub fn section(title: &str, body: &str) -> String {
    format!(
        r#"<section class="card"><h2>{}</h2>{}</section>"#,
        html_escape(title),
        body
    )
}

pub fn fmt_f(v: f64, decimals: usize) -> String {
    if v.is_nan() {
        return "—".into();
    }
    if v.is_infinite() {
        return "∞".into();
    }
    // Promote precision when the value is small enough that the requested
    // precision would round to zero (e.g. 0.022 with decimals=1 should show
    // as 0.02, not 0.0).
    if v != 0.0 && v.abs() < 10f64.powi(-(decimals as i32)) {
        let extra = (-(v.abs().log10().floor() as i32)).max(decimals as i32 + 1) as usize;
        return format!("{:.*}", extra.min(6), v);
    }
    format!("{:.*}", decimals, v)
}

/// Browser-friendly numeric value for `<input type="number">` round-trip.
///
/// Plain integer for whole numbers below 1e6; scientific for very large
/// or very small magnitudes; plain decimal otherwise.
pub fn fmt_num(v: f64) -> String {
    if !v.is_finite() || v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if v.fract() == 0.0 && a < 1e6 {
        return format!("{}", v as i64);
    }
    if a >= 1e5 || a < 1e-3 {
        // Compact scientific: trim trailing zeros from mantissa
        let s = format!("{:e}", v);
        return s;
    }
    let s = format!("{}", v);
    s
}

pub fn fmt_eng(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if a >= 1e12 {
        format!("{:.2}T", v / 1e12)
    } else if a >= 1e9 {
        format!("{:.2}G", v / 1e9)
    } else if a >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if a >= 1e3 {
        format!("{:.2}k", v / 1e3)
    } else if a >= 1.0 {
        format!("{:.3}", v)
    } else if a >= 1e-3 {
        format!("{:.4}", v)
    } else {
        format!("{:.3e}", v)
    }
}

pub fn fmt_usd(v: f64) -> String {
    let a = v.abs();
    if a >= 1e12 {
        format!("${:.2}T", v / 1e12)
    } else if a >= 1e9 {
        format!("${:.2}B", v / 1e9)
    } else if a >= 1e6 {
        format!("${:.2}M", v / 1e6)
    } else if a >= 1e3 {
        format!("${:.2}k", v / 1e3)
    } else if a >= 1.0 {
        format!("${:.2}", v)
    } else if a >= 1e-2 {
        format!("${:.4}", v)
    } else {
        format!("${:.6}", v)
    }
}

const STYLE: &str = r#"<style>
:root {
  --bg: #0a0c12;
  --panel: #11141d;
  --panel-2: #161a26;
  --line: #232735;
  --text: #e6e8ee;
  --muted: #9aa0b3;
  --cyan: #00d4ff;
  --purple: #a855f7;
  --green: #22c55e;
  --orange: #f59e0b;
  --pink: #ec4899;
  --red: #ef4444;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, sans-serif;
  font-size: 14px;
  line-height: 1.55;
}
a { color: var(--cyan); text-decoration: none; }
a:hover { text-decoration: underline; }

.hdr {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(10, 12, 18, 0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
}
.hdr-inner {
  max-width: 1200px; margin: 0 auto;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.75rem 1.25rem; gap: 1.25rem;
  flex-wrap: wrap;
}
.logo { font-weight: 700; font-size: 1rem; color: var(--text); }
.logo .muted { color: var(--muted); font-weight: 400; }
.nav { display: flex; gap: 0.25rem; flex-wrap: wrap; }
.nav-link {
  padding: 0.4rem 0.7rem;
  border-radius: 6px;
  color: var(--muted);
  font-size: 13px;
}
.nav-link:hover { background: var(--panel); color: var(--text); text-decoration: none; }
.nav-link.active { background: var(--panel-2); color: var(--cyan); }

.container { max-width: 1200px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }

h1 { font-size: 1.75rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.125rem; margin: 0 0 0.75rem; color: var(--cyan); font-weight: 600; }
h3 { font-size: 1rem; margin: 0 0 0.5rem; color: var(--text); }
.subtitle { color: var(--muted); margin: 0 0 1.5rem; }
.muted { color: var(--muted); }

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1.25rem;
  margin-bottom: 1rem;
}

.note { color: var(--muted); font-size: 13px; margin: 0.5rem 0 0; }

.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.75rem;
}
.field { display: flex; flex-direction: column; gap: 0.25rem; }
.field-label { color: var(--text); font-size: 12px; font-weight: 500; }
.field-hint { color: var(--muted); font-size: 11px; }
.field input, .field select {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.45rem 0.6rem;
  color: var(--text);
  font-family: ui-monospace, 'JetBrains Mono', monospace;
  font-size: 13px;
}
.field input:focus, .field select:focus {
  outline: none;
  border-color: var(--cyan);
  box-shadow: 0 0 0 1px var(--cyan);
}
.params-actions { margin-top: 0.75rem; display: flex; gap: 0.5rem; }
button {
  background: var(--cyan);
  color: #001520;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 0.9rem;
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
}
button:hover { filter: brightness(1.1); }
button[type="reset"] { background: var(--panel-2); color: var(--muted); }
button[type="reset"]:hover { background: var(--line); color: var(--text); }

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.stat {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.85rem;
}
.stat-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value {
  color: var(--green);
  font-family: ui-monospace, monospace;
  font-size: 1.4rem;
  font-weight: 700;
  margin: 0.2rem 0;
}
.stat-hint { color: var(--muted); font-size: 11px; }

.table-wrap { overflow-x: auto; }
table.data {
  width: 100%; border-collapse: collapse;
  font-family: ui-monospace, monospace;
  font-size: 13px;
}
table.data th, table.data td {
  padding: 0.5rem 0.75rem;
  text-align: right;
  border-bottom: 1px solid var(--line);
}
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
table.data tr:hover td { background: var(--panel-2); }

.chart svg { width: 100%; height: auto; max-width: 100%; border-radius: 8px; }

.tile-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.75rem;
}
.tile {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem;
}
.tile h3 a { color: var(--cyan); }
.tile p { margin: 0.4rem 0 0; color: var(--muted); font-size: 13px; }

.ftr { text-align: center; color: var(--muted); padding: 1.5rem; font-size: 12px; }
</style>"#;
