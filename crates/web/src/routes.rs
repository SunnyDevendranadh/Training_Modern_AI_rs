//! HTTP routes — every page is server-rendered HTML.
//!
//! All physics computation runs server-side. Forms submit via GET so URLs
//! are shareable. The client receives a complete HTML document with embedded
//! SVG charts; no JavaScript required.

use axum::{extract::Query, response::Html, routing::get, Json, Router};
use ndarray::Array1;
use physics::*;
use serde::Deserialize;

use crate::chart::{Axis, BarChart, LineChart, Marker, Scale, Series};
use crate::render::{
    fmt_eng, fmt_f, fmt_num, fmt_usd, form, page, section, stats_grid, table, Field, FieldKind,
    Stat,
};

pub fn create_router() -> Router {
    Router::new()
        .route("/", get(home))
        .route("/roofline", get(roofline))
        .route("/cost", get(cost))
        .route("/context", get(context))
        .route("/agents-md", get(agents_md))
        .route("/coordination", get(coordination))
        .route("/scaling", get(scaling))
        .route("/pricing", get(pricing))
        .route("/throughput", get(throughput))
        .route("/api/health", get(health))
}

// =====================================================================
// Home
// =====================================================================

async fn home() -> Html<String> {
    let intro = r#"
<h1>Training Modern AI</h1>
<p class="subtitle">Interactive economics of frontier transformer inference, end-to-end in Rust. Every parameter is editable; every result is computed on the server.</p>
"#;

    let mut tiles = String::from(r#"<div class="tile-grid">"#);
    for (path, label, desc) in crate::render::NAV.iter().skip(1) {
        tiles.push_str(&format!(
            r#"<div class="tile"><h3><a href="{p}">{l} →</a></h3><p>{d}</p></div>"#,
            p = path,
            l = crate::chart::html_escape(label),
            d = crate::chart::html_escape(desc),
        ));
    }
    tiles.push_str("</div>");

    let insights = r#"
<ul style="margin: 0; padding-left: 1.25rem; line-height: 1.9;">
  <li><strong>Latency floor ≈ 20 ms</strong> — physics, not engineering. Set by weight fetch from HBM.</li>
  <li><strong>Batch sweet spot ≈ 2,000</strong> sequences for modern sparse MoE models.</li>
  <li><strong>Rack boundary</strong> bounds expert parallelism — one NVLink domain per rack.</li>
  <li><strong>Context wall is bandwidth, not compute</strong> — KV cache reads dominate at long context.</li>
  <li><strong>~100× over-training</strong> vs Chinchilla — driven by inference demand amortization.</li>
  <li><strong>Agent inference ≈ 10–50× a chat turn</strong> — multi-step harness loops.</li>
  <li><strong>AGENTS.md ≈ 8 KB</strong> — Vercel’s empirical sweet spot for passive context.</li>
  <li><strong>Recursive Planner+Worker</strong> — the only multi-agent pattern that scales near-linearly.</li>
</ul>
"#;

    let body = format!(
        "{intro}{tiles_section}{insights_section}",
        tiles_section = section("Modules", &tiles),
        insights_section = section("Key insights", insights),
    );

    Html(page("Home", "/", &body))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "tests": 85 }))
}

// =====================================================================
// Roofline
// =====================================================================

#[derive(Deserialize)]
struct RooflineParams {
    n_active: Option<f64>,
    n_total: Option<f64>,
    context: Option<f64>,
    bytes_per_param: Option<f64>,
    flops: Option<f64>,
    mem_bw: Option<f64>,
    bytes_per_token: Option<f64>,
}

async fn roofline(Query(p): Query<RooflineParams>) -> Html<String> {
    let n_active = p.n_active.unwrap_or(N_ACTIVE);
    let n_total = p.n_total.unwrap_or(N_TOTAL);
    let ctx_len = p.context.unwrap_or(CONTEXT_LENGTH);
    let bpp = p.bytes_per_param.unwrap_or(BYTES_PER_PARAM_FP8);
    let flops_v = p.flops.unwrap_or(FLOPS);
    let mem_bw = p.mem_bw.unwrap_or(MEM_BW);
    let bpt = p.bytes_per_token.unwrap_or(BYTES_PER_TOKEN);

    let bs = Array1::logspace(10.0, 0.0, 4.5, 200);
    let ctx_arr = Array1::from_elem(200, ctx_len);
    let (total, tc, tw, tk) =
        total_latency(&bs, n_active, n_total, &ctx_arr, bpp, bpt, flops_v, mem_bw);

    let (idx, balance_bs, _) =
        physics::latency::find_balance_point(&bs, n_active, n_total, &ctx_arr, bpp, bpt, flops_v, mem_bw);
    let balance_lat_ms = total[idx] * 1000.0;
    let lat_floor_ms = total[0] * 1000.0;
    let weights_ms = tw[0] * 1000.0;

    let stats = stats_grid(&[
        Stat::new("Balance batch size", fmt_eng(balance_bs), "compute time = memory time"),
        Stat::new("Latency at balance", format!("{} ms", fmt_f(balance_lat_ms, 1)), "Min cost-per-token regime"),
        Stat::new("Latency floor (B=1)", format!("{} ms", fmt_f(lat_floor_ms, 1)), "Memory-bound; physics, not engineering"),
        Stat::new("Weight fetch", format!("{} ms", fmt_f(weights_ms, 1)), "N_total × bytes/param ÷ bandwidth"),
    ]);

    // Chart
    let series: Vec<(f64, f64)> = bs.iter().zip(total.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();
    let s_compute: Vec<(f64, f64)> = bs.iter().zip(tc.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();
    let s_weights: Vec<(f64, f64)> = bs.iter().zip(tw.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();
    let s_kv: Vec<(f64, f64)> = bs.iter().zip(tk.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();

    let bal_label = format!("balance≈{}", fmt_eng(balance_bs));
    let chart = LineChart {
        title: "Latency vs batch size (log–log)",
        x_label: "Batch size (sequences)",
        y_label: "Latency (ms)",
        x_scale: Scale::Log10,
        y_scale: Scale::Log10,
        series: vec![
            Series { name: "Total",   color: "#00d4ff", points: series },
            Series { name: "Compute", color: "#22c55e", points: s_compute },
            Series { name: "Weights", color: "#a855f7", points: s_weights },
            Series { name: "KV cache",color: "#f59e0b", points: s_kv },
        ],
        markers: vec![Marker {
            axis: Axis::X,
            value: balance_bs,
            label: &bal_label,
            color: "#ec4899",
        }],
    }
    .render();

    // Sample table
    let sample_bs = [1.0, 32.0, 256.0, 2048.0, 8192.0, 32768.0];
    let mut rows = vec![];
    for &b in &sample_bs {
        let bb = Array1::from_elem(1, b);
        let cc = Array1::from_elem(1, ctx_len);
        let (tt, ttc, ttw, ttk) =
            total_latency(&bb, n_active, n_total, &cc, bpp, bpt, flops_v, mem_bw);
        rows.push(vec![
            fmt_eng(b),
            format!("{} ms", fmt_f(tt[0] * 1000.0, 2)),
            format!("{} ms", fmt_f(ttc[0] * 1000.0, 2)),
            format!("{} ms", fmt_f(ttw[0] * 1000.0, 2)),
            format!("{} ms", fmt_f(ttk[0] * 1000.0, 2)),
            if ttc[0] > ttw[0] + ttk[0] { "compute".into() } else { "memory".into() },
        ]);
    }
    let tbl = table(
        &["Batch", "Total", "Compute", "Weights", "KV", "Bottleneck"],
        &rows,
    );

    let form_html = form(
        "/roofline",
        &[
            Field {
                name: "n_active",
                label: "Active params (per token)",
                hint: "Default 37e9 (37B for sparse MoE)",
                value: fmt_num(n_active),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") },
            },
            Field {
                name: "n_total",
                label: "Total params",
                hint: "Default 700e9 (full MoE)",
                value: fmt_num(n_total),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") },
            },
            Field {
                name: "context",
                label: "Context length (tokens)",
                hint: "Default 32,768",
                value: fmt_f(ctx_len, 0),
                kind: FieldKind::Number { step: "1024", min: Some("1") },
            },
            Field {
                name: "bytes_per_param",
                label: "Bytes per param",
                hint: "FP8 = 1.0, FP4 = 0.5",
                value: fmt_f(bpp, 1),
                kind: FieldKind::Select {
                    options: &[("0.5", "FP4 (0.5)"), ("1.0", "FP8 (1.0)"), ("2.0", "FP16 (2.0)")],
                },
            },
            Field {
                name: "flops",
                label: "FLOPS (rack aggregate)",
                hint: "Default 1.5e15 (Blackwell-NVL72)",
                value: fmt_num(flops_v),
                kind: FieldKind::Number { step: "1e14", min: Some("1e12") },
            },
            Field {
                name: "mem_bw",
                label: "Memory bandwidth (B/s)",
                hint: "Default 5e12 (HBM aggregate)",
                value: fmt_num(mem_bw),
                kind: FieldKind::Number { step: "1e11", min: Some("1e10") },
            },
            Field {
                name: "bytes_per_token",
                label: "KV bytes / token",
                hint: "Default 2048",
                value: fmt_f(bpt, 0),
                kind: FieldKind::Number { step: "256", min: Some("1") },
            },
        ],
    );

    let body = format!(
        r#"<h1>Roofline analysis</h1>
<p class="subtitle">Total latency = max(compute, weights + KV cache). Below the balance batch you are memory-bound; above it you are compute-bound.</p>
{form}
{stats}
{chart_section}
{table_section}"#,
        form = section("Parameters", &form_html),
        stats = stats,
        chart_section = section("Latency decomposition", &format!(r#"<div class="chart">{}</div>"#, chart)),
        table_section = section("Sample batch sizes", &tbl),
    );

    Html(page("Roofline", "/roofline", &body))
}

// =====================================================================
// Cost
// =====================================================================

#[derive(Deserialize)]
struct CostParams {
    n_active: Option<f64>,
    n_total: Option<f64>,
    context: Option<f64>,
    bytes_per_param: Option<f64>,
    gpu_cost_per_hour: Option<f64>,
    gpus_in_rack: Option<usize>,
}

async fn cost(Query(p): Query<CostParams>) -> Html<String> {
    let n_active = p.n_active.unwrap_or(N_ACTIVE);
    let n_total = p.n_total.unwrap_or(N_TOTAL);
    let ctx_len = p.context.unwrap_or(CONTEXT_LENGTH);
    let bpp = p.bytes_per_param.unwrap_or(BYTES_PER_PARAM_FP8);
    let gph = p.gpu_cost_per_hour.unwrap_or(GPU_COST_PER_HOUR);
    let gpus = p.gpus_in_rack.unwrap_or(GPUS_IN_RACK);

    let rc = rack_cost_per_sec(gph, gpus);
    let bs = Array1::logspace(10.0, 0.0, 5.0, 300);
    let ctx_arr = Array1::from_elem(300, ctx_len);
    let cost_arr = cost_per_million_tokens(
        &bs, n_active, n_total, &ctx_arr, bpp, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
    );
    let (total_lat, _, _, _) =
        total_latency(&bs, n_active, n_total, &ctx_arr, bpp, BYTES_PER_TOKEN, FLOPS, MEM_BW);

    let (min_idx, _) = cost_arr
        .indexed_iter()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let opt_batch = bs[min_idx];
    let opt_cost = cost_arr[min_idx];
    let opt_lat = total_lat[min_idx] * 1000.0;
    let floor = physics::cost::compute_cost_floor(n_active, FLOPS, rc);

    let stats = stats_grid(&[
        Stat::new("Cost-min batch", fmt_eng(opt_batch), "Where $/M tokens bottoms out"),
        Stat::new("Cost at min", fmt_usd(opt_cost), "$ per million tokens"),
        Stat::new("Latency at min", format!("{} ms", fmt_f(opt_lat, 1)), "Latency you pay for the min cost"),
        Stat::new("Compute floor", fmt_usd(floor), "Asymptotic cost at infinite batch"),
    ]);

    let cost_pts: Vec<(f64, f64)> =
        bs.iter().zip(cost_arr.iter()).map(|(&x, &y)| (x, y)).collect();
    let chart = LineChart {
        title: "Cost vs batch size (log–log)",
        x_label: "Batch size",
        y_label: "$ / million tokens",
        x_scale: Scale::Log10,
        y_scale: Scale::Log10,
        series: vec![Series {
            name: "Cost / M tokens",
            color: "#00d4ff",
            points: cost_pts,
        }],
        markers: vec![Marker {
            axis: Axis::Y,
            value: floor,
            label: "compute floor",
            color: "#22c55e",
        }],
    }
    .render();

    let mut rows = vec![];
    for b in [1.0, 32.0, 256.0, 2048.0, 8192.0, 32768.0] {
        let bb = Array1::from_elem(1, b);
        let cc = Array1::from_elem(1, ctx_len);
        let cv = cost_per_million_tokens(
            &bb, n_active, n_total, &cc, bpp, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );
        let (lt, _, _, _) =
            total_latency(&bb, n_active, n_total, &cc, bpp, BYTES_PER_TOKEN, FLOPS, MEM_BW);
        rows.push(vec![
            fmt_eng(b),
            fmt_usd(cv[0]),
            format!("{} ms", fmt_f(lt[0] * 1000.0, 2)),
            format!("{}× floor", fmt_f(cv[0] / floor.max(1e-12), 1)),
        ]);
    }
    let tbl = table(&["Batch", "Cost / M tok", "Latency", "Multiple of floor"], &rows);

    let form_html = form(
        "/cost",
        &[
            Field {
                name: "n_active",
                label: "Active params",
                hint: "Default 37e9",
                value: fmt_num(n_active),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") },
            },
            Field {
                name: "n_total",
                label: "Total params",
                hint: "Default 700e9",
                value: fmt_num(n_total),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") },
            },
            Field {
                name: "context",
                label: "Context length",
                hint: "Default 32,768",
                value: fmt_f(ctx_len, 0),
                kind: FieldKind::Number { step: "1024", min: Some("1") },
            },
            Field {
                name: "bytes_per_param",
                label: "Bytes per param",
                hint: "FP8 = 1.0",
                value: fmt_f(bpp, 1),
                kind: FieldKind::Select {
                    options: &[("0.5", "FP4 (0.5)"), ("1.0", "FP8 (1.0)"), ("2.0", "FP16 (2.0)")],
                },
            },
            Field {
                name: "gpu_cost_per_hour",
                label: "GPU $/hour",
                hint: "Default $2.00",
                value: fmt_f(gph, 2),
                kind: FieldKind::Number { step: "0.10", min: Some("0.01") },
            },
            Field {
                name: "gpus_in_rack",
                label: "GPUs in rack",
                hint: "Default 72 (NVL72)",
                value: gpus.to_string(),
                kind: FieldKind::Number { step: "1", min: Some("1") },
            },
        ],
    );

    let body = format!(
        r#"<h1>Cost per million tokens</h1>
<p class="subtitle">Bigger batches amortize the rack’s fixed cost over more tokens. Cost approaches a compute floor — never zero.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Cost curve", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Sample batch sizes", &tbl),
    );

    Html(page("Cost", "/cost", &body))
}

// =====================================================================
// Context wall
// =====================================================================

#[derive(Deserialize)]
struct ContextParams {
    batch: Option<f64>,
    n_active: Option<f64>,
    n_total: Option<f64>,
    bytes_per_param: Option<f64>,
    bytes_per_token: Option<f64>,
}

async fn context(Query(p): Query<ContextParams>) -> Html<String> {
    let batch = p.batch.unwrap_or(2000.0);
    let n_active = p.n_active.unwrap_or(N_ACTIVE);
    let n_total = p.n_total.unwrap_or(N_TOTAL);
    let bpp = p.bytes_per_param.unwrap_or(BYTES_PER_PARAM_FP8);
    let bpt = p.bytes_per_token.unwrap_or(BYTES_PER_TOKEN);

    let ctx = Array1::logspace(10.0, 1.0, 6.0, 300);
    let bs = Array1::from_elem(300, batch);
    let tc = t_compute(&bs, n_active, FLOPS);
    let tw = t_mem_weights(&bs, n_total, bpp, MEM_BW);
    let tk = t_mem_kv(&bs, &ctx, bpt, MEM_BW);
    let tm = &tw + &tk;
    let total: Vec<f64> = (0..300).map(|i| tc[i].max(tm[i])).collect();

    // Find crossover where compute = memory (ignoring weights since they're constant)
    let cross_idx = (0..300)
        .min_by(|&i, &j| (tc[i] - tm[i]).abs().partial_cmp(&(tc[j] - tm[j]).abs()).unwrap())
        .unwrap();
    let crossover = ctx[cross_idx];

    let lat_at = |c: f64| -> f64 {
        let bb = Array1::from_elem(1, batch);
        let cc = Array1::from_elem(1, c);
        let (lt, _, _, _) = total_latency(&bb, n_active, n_total, &cc, bpp, bpt, FLOPS, MEM_BW);
        lt[0] * 1000.0
    };
    let stats = stats_grid(&[
        Stat::new("Crossover context", fmt_eng(crossover), "compute time = memory time"),
        Stat::new("Latency @ 32k", format!("{} ms", fmt_f(lat_at(32_768.0), 1)), "At given batch"),
        Stat::new("Latency @ 1M", format!("{} ms", fmt_f(lat_at(1_000_000.0), 1)), "Memory-bandwidth wall"),
        Stat::new("Batch (fixed)", fmt_eng(batch), "Sequences per forward"),
    ]);

    let s_total: Vec<(f64, f64)> = ctx.iter().zip(total.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();
    let s_compute: Vec<(f64, f64)> = ctx.iter().zip(tc.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();
    let s_kv: Vec<(f64, f64)> = ctx.iter().zip(tk.iter()).map(|(&x, &y)| (x, y * 1000.0)).collect();

    let cross_label = format!("crossover≈{}", fmt_eng(crossover));
    let chart = LineChart {
        title: "Latency vs context length",
        x_label: "Context length (tokens)",
        y_label: "Latency (ms)",
        x_scale: Scale::Log10,
        y_scale: Scale::Log10,
        series: vec![
            Series { name: "Total",   color: "#00d4ff", points: s_total },
            Series { name: "Compute", color: "#22c55e", points: s_compute },
            Series { name: "KV",      color: "#f59e0b", points: s_kv },
        ],
        markers: vec![Marker { axis: Axis::X, value: crossover, label: &cross_label, color: "#ec4899" }],
    }
    .render();

    let mut rows = vec![];
    for c in [1024.0, 8192.0, 32768.0, 131072.0, 1_000_000.0] {
        let bb = Array1::from_elem(1, batch);
        let cc = Array1::from_elem(1, c);
        let (tt, _, _, kk) = total_latency(&bb, n_active, n_total, &cc, bpp, bpt, FLOPS, MEM_BW);
        rows.push(vec![
            fmt_eng(c),
            format!("{} ms", fmt_f(tt[0] * 1000.0, 2)),
            format!("{} ms", fmt_f(kk[0] * 1000.0, 2)),
            format!("{:.1}×", kk[0] / tc[0].max(1e-12)),
        ]);
    }
    let tbl = table(&["Context", "Total latency", "KV time", "KV / compute"], &rows);

    let form_html = form(
        "/context",
        &[
            Field { name: "batch", label: "Batch size", hint: "Default 2000", value: fmt_f(batch, 0),
                kind: FieldKind::Number { step: "100", min: Some("1") } },
            Field { name: "n_active", label: "Active params", hint: "Default 37e9", value: fmt_num(n_active),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") } },
            Field { name: "n_total", label: "Total params", hint: "Default 700e9", value: fmt_num(n_total),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") } },
            Field { name: "bytes_per_param", label: "Bytes per param", hint: "FP8 = 1", value: fmt_f(bpp, 1),
                kind: FieldKind::Select {
                    options: &[("0.5", "FP4 (0.5)"), ("1.0", "FP8 (1.0)"), ("2.0", "FP16 (2.0)")],
                } },
            Field { name: "bytes_per_token", label: "KV bytes / token", hint: "Default 2048", value: fmt_f(bpt, 0),
                kind: FieldKind::Number { step: "256", min: Some("1") } },
        ],
    );

    let body = format!(
        r#"<h1>Context length: bandwidth wall</h1>
<p class="subtitle">At long context, KV cache reads dominate compute — the wall is memory bandwidth, not FLOPs.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Latency vs context", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Sample contexts", &tbl),
    );

    Html(page("Context", "/context", &body))
}

// =====================================================================
// AGENTS.md
// =====================================================================

#[derive(Deserialize)]
struct AgentsMdParams {
    optimal_kb: Option<f64>,
    sigma: Option<f64>,
    context_window: Option<f64>,
    tokens_per_kb: Option<f64>,
    task_complexity: Option<f64>,
}

async fn agents_md(Query(p): Query<AgentsMdParams>) -> Html<String> {
    let optimal = p.optimal_kb.unwrap_or(8.0);
    let sigma = p.sigma.unwrap_or(1.2);
    let ctx_window = p.context_window.unwrap_or(128_000.0);
    let tokens_kb = p.tokens_per_kb.unwrap_or(150.0);
    let task = p.task_complexity.unwrap_or(5000.0);

    let sizes: Vec<f64> = (0..200)
        .map(|i| 10.0_f64.powf(-1.0 + 3.0 * i as f64 / 199.0))
        .collect();
    let passive: Vec<f64> = sizes.iter()
        .map(|&s| physics::knowledge::agents_md_passive_effectiveness(s, optimal, sigma, ctx_window, tokens_kb, task))
        .collect();
    let active: Vec<f64> = sizes.iter()
        .map(|&s| physics::knowledge::agents_md_active_effectiveness(s, optimal, sigma * 1.5))
        .collect();

    let (best_i, &best_v) = passive.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let best_size = sizes[best_i];

    let pollution = physics::knowledge::agents_md_passive_effectiveness(
        500.0, optimal, sigma, ctx_window, tokens_kb, task,
    );
    let stats = stats_grid(&[
        Stat::new("Optimal size", format!("{} KB", fmt_f(best_size, 2)), "Maximum passive effectiveness"),
        Stat::new("Max success", format!("{:.0}%", best_v), "Passive (auto-injected) context"),
        Stat::new("Active ceiling", format!("{:.0}%", active.iter().cloned().fold(0.0_f64, f64::max)), "Skills-based retrieval (Vercel: ~79%)"),
        Stat::new("Pollution at 500 KB", format!("{:.0}%", pollution), "Context window flooded"),
    ]);

    let s_passive: Vec<(f64, f64)> = sizes.iter().zip(passive.iter()).map(|(&x, &y)| (x, y)).collect();
    let s_active: Vec<(f64, f64)> = sizes.iter().zip(active.iter()).map(|(&x, &y)| (x, y)).collect();

    let opt_label = format!("optimal≈{:.1}KB", best_size);
    let chart = LineChart {
        title: "AGENTS.md effectiveness",
        x_label: "Size (KB)",
        y_label: "Success rate (%)",
        x_scale: Scale::Log10,
        y_scale: Scale::Linear,
        series: vec![
            Series { name: "Passive", color: "#00d4ff", points: s_passive },
            Series { name: "Active",  color: "#a855f7", points: s_active },
        ],
        markers: vec![Marker { axis: Axis::X, value: best_size, label: &opt_label, color: "#22c55e" }],
    }
    .render();

    let mut rows = vec![];
    for sz in [0.5, 2.0, 8.0, 16.0, 50.0, 200.0, 500.0] {
        rows.push(vec![
            format!("{} KB", fmt_f(sz, 1)),
            format!("{:.1}%", physics::knowledge::agents_md_passive_effectiveness(sz, optimal, sigma, ctx_window, tokens_kb, task)),
            format!("{:.1}%", physics::knowledge::agents_md_active_effectiveness(sz, optimal, sigma * 1.5)),
            format!("{:.0}", sz * tokens_kb),
        ]);
    }
    let tbl = table(&["Size", "Passive", "Active", "Tokens used"], &rows);

    let form_html = form(
        "/agents-md",
        &[
            Field { name: "optimal_kb", label: "Optimal size (KB)", hint: "Vercel: ~8 KB", value: fmt_f(optimal, 1),
                kind: FieldKind::Number { step: "0.5", min: Some("0.1") } },
            Field { name: "sigma", label: "Sigma (KB)", hint: "Width of effectiveness peak", value: fmt_f(sigma, 2),
                kind: FieldKind::Number { step: "0.1", min: Some("0.1") } },
            Field { name: "context_window", label: "Context window (tokens)", hint: "e.g. 128k = 128000", value: fmt_f(ctx_window, 0),
                kind: FieldKind::Number { step: "1000", min: Some("1000") } },
            Field { name: "tokens_per_kb", label: "Tokens per KB", hint: "Markdown ≈ 150", value: fmt_f(tokens_kb, 0),
                kind: FieldKind::Number { step: "10", min: Some("1") } },
            Field { name: "task_complexity", label: "Task complexity (tokens)", hint: "Tokens already in context", value: fmt_f(task, 0),
                kind: FieldKind::Number { step: "500", min: Some("0") } },
        ],
    );

    let body = format!(
        r#"<h1>AGENTS.md effectiveness</h1>
<p class="subtitle">Passive context (auto-injected) hits 100% near the sweet spot but pollutes the window when large. Active retrieval caps at ~79% but degrades gracefully.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Effectiveness curve", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Sample sizes", &tbl),
    );

    Html(page("AGENTS.md", "/agents-md", &body))
}

// =====================================================================
// Coordination
// =====================================================================

#[derive(Deserialize)]
struct CoordParams {
    max_agents: Option<usize>,
}

async fn coordination(Query(p): Query<CoordParams>) -> Html<String> {
    use physics::agents::effective_throughput;
    use physics::types::CoordinationStrategy;

    let max = p.max_agents.unwrap_or(100).clamp(2, 500);

    let xs: Vec<f64> = (1..=max).map(|n| n as f64).collect();
    let s_eq: Vec<f64> = (1..=max).map(|n| effective_throughput(n, CoordinationStrategy::Equal)).collect();
    let s_pi: Vec<f64> = (1..=max).map(|n| effective_throughput(n, CoordinationStrategy::Pipeline)).collect();
    let s_co: Vec<f64> = (1..=max).map(|n| effective_throughput(n, CoordinationStrategy::Continuous)).collect();
    let s_re: Vec<f64> = (1..=max).map(|n| effective_throughput(n, CoordinationStrategy::Recursive)).collect();

    let r_max = effective_throughput(max, CoordinationStrategy::Recursive);
    let e_max = effective_throughput(max, CoordinationStrategy::Equal);
    let p_max = effective_throughput(max, CoordinationStrategy::Pipeline);
    let c_max = effective_throughput(max, CoordinationStrategy::Continuous);

    let stats = stats_grid(&[
        Stat::new(format!("Recursive @ {}", max), fmt_f(r_max, 1), "Planner+Worker — wins"),
        Stat::new(format!("Pipeline @ {}", max), fmt_f(p_max, 1), "Bottlenecked by slowest stage"),
        Stat::new(format!("Continuous @ {}", max), fmt_f(c_max, 1), "Regresses past N≈10"),
        Stat::new(format!("Equal @ {}", max), fmt_f(e_max, 1), "Lock contention collapse"),
    ]);

    let pts = |v: &[f64]| -> Vec<(f64, f64)> {
        xs.iter().zip(v.iter()).map(|(&x, &y)| (x, y)).collect()
    };
    let chart = LineChart {
        title: "Effective throughput vs agent count",
        x_label: "Number of agents",
        y_label: "Effective parallel agents",
        x_scale: Scale::Linear,
        y_scale: Scale::Linear,
        series: vec![
            Series { name: "Recursive",  color: "#22c55e", points: pts(&s_re) },
            Series { name: "Pipeline",   color: "#a855f7", points: pts(&s_pi) },
            Series { name: "Continuous", color: "#f59e0b", points: pts(&s_co) },
            Series { name: "Equal",      color: "#ef4444", points: pts(&s_eq) },
        ],
        markers: vec![],
    }
    .render();

    let mut rows = vec![];
    for n in [1usize, 5, 10, 20, 50, 100] {
        if n > max { continue; }
        rows.push(vec![
            n.to_string(),
            fmt_f(effective_throughput(n, CoordinationStrategy::Recursive), 2),
            fmt_f(effective_throughput(n, CoordinationStrategy::Pipeline), 2),
            fmt_f(effective_throughput(n, CoordinationStrategy::Continuous), 2),
            fmt_f(effective_throughput(n, CoordinationStrategy::Equal), 2),
        ]);
    }
    let tbl = table(&["N agents", "Recursive", "Pipeline", "Continuous", "Equal"], &rows);

    let form_html = form(
        "/coordination",
        &[Field {
            name: "max_agents",
            label: "Max agents to plot",
            hint: "Default 100",
            value: max.to_string(),
            kind: FieldKind::Number { step: "10", min: Some("2") },
        }],
    );

    let body = format!(
        r#"<h1>Multi-agent coordination</h1>
<p class="subtitle">Cursor's four iterations. Only Recursive Planner+Worker scales near-linearly — the others all collapse or saturate.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Coordination strategies", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Sample sizes", &tbl),
    );

    Html(page("Coordination", "/coordination", &body))
}

// =====================================================================
// Scaling laws
// =====================================================================

#[derive(Deserialize)]
struct ScalingParams {
    n_active: Option<f64>,
    tokens_per_sec: Option<f64>,
    months: Option<f64>,
    alpha_rl: Option<f64>,
    cost_per_flop: Option<f64>,
}

async fn scaling(Query(p): Query<ScalingParams>) -> Html<String> {
    use physics::scaling::{
        inference_tokens_served, optimal_pretrain_ratio, over_training_factor, total_cost,
        FlopsPerToken,
    };

    let n_active = p.n_active.unwrap_or(100e9);
    let tps = p.tokens_per_sec.unwrap_or(50e6);
    let months = p.months.unwrap_or(2.0);
    let alpha = p.alpha_rl.unwrap_or(0.5);
    let cpf = p.cost_per_flop.unwrap_or(1e-15);

    let d_inf = inference_tokens_served(tps, months);
    let ratios = Array1::logspace(10.0, -2.0, 3.0, 400);
    let fpt = FlopsPerToken::default();

    let (best_idx, best_ratio) =
        optimal_pretrain_ratio(d_inf, n_active, &fpt, cpf, alpha, &ratios);
    let pretrain_tokens = best_ratio * d_inf;
    let over = over_training_factor(pretrain_tokens, n_active);

    let mut series_total = Vec::new();
    let mut series_pt = Vec::new();
    let mut series_inf = Vec::new();
    for &r in ratios.iter() {
        let d_pt = r * d_inf;
        let (t, pt, _, inf) = total_cost(d_pt, d_pt, d_inf, n_active, &fpt, cpf, alpha);
        series_total.push((r, t));
        series_pt.push((r, pt));
        series_inf.push((r, inf));
    }

    let (best_total, best_pt, best_rl, best_inf) = total_cost(
        best_ratio * d_inf, best_ratio * d_inf, d_inf, n_active, &fpt, cpf, alpha,
    );

    let stats = stats_grid(&[
        Stat::new("Inference tokens", fmt_eng(d_inf), format!("{:.0}M tok/s × {:.1} mo", tps / 1e6, months)),
        Stat::new("Optimal ratio", fmt_f(best_ratio, 3), "pretrain : inference tokens"),
        Stat::new("Over-training factor", format!("{:.1}×", over), "vs Chinchilla (20×N)"),
        Stat::new("Total cost", fmt_usd(best_total), "at optimal ratio"),
    ]);

    let chart = LineChart {
        title: "Total cost vs pretrain:inference ratio",
        x_label: "Ratio (D_pretrain / D_inference)",
        y_label: "$ cost",
        x_scale: Scale::Log10,
        y_scale: Scale::Log10,
        series: vec![
            Series { name: "Total",     color: "#00d4ff", points: series_total },
            Series { name: "Pretrain",  color: "#a855f7", points: series_pt },
            Series { name: "Inference", color: "#22c55e", points: series_inf },
        ],
        markers: vec![Marker {
            axis: Axis::X,
            value: best_ratio,
            label: "optimum",
            color: "#ec4899",
        }],
    }
    .render();

    let rows = vec![
        vec!["Pretrain".into(), fmt_usd(best_pt), format!("{:.1}%", best_pt / best_total * 100.0)],
        vec!["RL".into(),       fmt_usd(best_rl), format!("{:.1}%", best_rl / best_total * 100.0)],
        vec!["Inference".into(),fmt_usd(best_inf),format!("{:.1}%", best_inf / best_total * 100.0)],
        vec!["Total".into(),    fmt_usd(best_total), "100.0%".into()],
    ];
    let tbl = table(&["Stage", "Cost", "Share"], &rows);
    let _ = best_idx;

    let form_html = form(
        "/scaling",
        &[
            Field { name: "n_active", label: "Active params", hint: "100B for frontier", value: fmt_num(n_active),
                kind: FieldKind::Number { step: "1e9", min: Some("1e6") } },
            Field { name: "tokens_per_sec", label: "Tokens/sec served", hint: "Default 50M", value: fmt_num(tps),
                kind: FieldKind::Number { step: "1e6", min: Some("1") } },
            Field { name: "months", label: "Months running", hint: "Default 2", value: fmt_f(months, 1),
                kind: FieldKind::Number { step: "0.5", min: Some("0.1") } },
            Field { name: "alpha_rl", label: "RL inefficiency α", hint: "Default 0.5", value: fmt_f(alpha, 2),
                kind: FieldKind::Number { step: "0.1", min: Some("0") } },
            Field { name: "cost_per_flop", label: "Cost per FLOP", hint: "Default 1e-15", value: fmt_num(cpf),
                kind: FieldKind::Number { step: "1e-16", min: Some("1e-20") } },
        ],
    );

    let body = format!(
        r#"<h1>Scaling laws & over-training</h1>
<p class="subtitle">Frontier models over-train ~100× vs Chinchilla. Inference demand makes pretraining cheap to amortize.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Cost vs ratio", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Cost breakdown at optimum", &tbl),
    );
    Html(page("Scaling", "/scaling", &body))
}

// =====================================================================
// Pricing
// =====================================================================

#[derive(Deserialize)]
struct PricingParams {
    sandbox_per_hour: Option<f64>,
    anthropic_per_hour: Option<f64>,
    tokens_per_hour: Option<f64>,
    model_per_million: Option<f64>,
    max_hours: Option<f64>,
}

async fn pricing(Query(p): Query<PricingParams>) -> Html<String> {
    use physics::pricing::{anthropic_total_cost, find_break_even, openai_total_cost};

    let oa_h = p.sandbox_per_hour.unwrap_or(0.50);
    let an_h = p.anthropic_per_hour.unwrap_or(0.08);
    let tph = p.tokens_per_hour.unwrap_or(50_000.0);
    let mpm = p.model_per_million.unwrap_or(0.50);
    let max_h = p.max_hours.unwrap_or(1000.0).clamp(10.0, 1_000_000.0);

    let n = 200;
    let hours: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powf((max_h.log10() - 0.0) * i as f64 / (n - 1) as f64))
        .collect();
    let oa_costs: Vec<f64> = hours.iter().map(|&h| openai_total_cost(h, oa_h, tph, mpm)).collect();
    let an_costs: Vec<f64> = hours.iter().map(|&h| anthropic_total_cost(h, an_h, tph, mpm)).collect();

    let break_even = find_break_even(&hours, &oa_costs, &an_costs);

    let stats = stats_grid(&[
        Stat::new("OpenAI @ 100h", fmt_usd(openai_total_cost(100.0, oa_h, tph, mpm)), "sandbox + model"),
        Stat::new("Anthropic @ 100h", fmt_usd(anthropic_total_cost(100.0, an_h, tph, mpm)), "harness + model"),
        Stat::new("Break-even", break_even.map(|h| format!("{} h", fmt_f(h, 1))).unwrap_or_else(|| "none in range".into()), "Where curves cross"),
        Stat::new("Model $/M tokens", fmt_usd(mpm), "Inference rate"),
    ]);

    let chart = LineChart {
        title: "Cumulative session cost",
        x_label: "Session hours",
        y_label: "$ total",
        x_scale: Scale::Log10,
        y_scale: Scale::Log10,
        series: vec![
            Series { name: "OpenAI (sandbox)", color: "#00d4ff",
                     points: hours.iter().zip(oa_costs.iter()).map(|(&x,&y)| (x,y)).collect() },
            Series { name: "Anthropic (MCP)",  color: "#a855f7",
                     points: hours.iter().zip(an_costs.iter()).map(|(&x,&y)| (x,y)).collect() },
        ],
        markers: vec![],
    }
    .render();

    let mut rows = vec![];
    for h in [1.0, 10.0, 100.0, 1000.0, 10000.0] {
        if h > max_h { continue; }
        rows.push(vec![
            fmt_f(h, 0),
            fmt_usd(openai_total_cost(h, oa_h, tph, mpm)),
            fmt_usd(anthropic_total_cost(h, an_h, tph, mpm)),
            fmt_f(openai_total_cost(h, oa_h, tph, mpm) / anthropic_total_cost(h, an_h, tph, mpm).max(1e-12), 2),
        ]);
    }
    let tbl = table(&["Hours", "OpenAI", "Anthropic", "OA / Anth"], &rows);

    let form_html = form(
        "/pricing",
        &[
            Field { name: "sandbox_per_hour", label: "OpenAI sandbox $/h", hint: "Default 0.50", value: fmt_f(oa_h, 2),
                kind: FieldKind::Number { step: "0.05", min: Some("0") } },
            Field { name: "anthropic_per_hour", label: "Anthropic harness $/h", hint: "Default 0.08", value: fmt_f(an_h, 2),
                kind: FieldKind::Number { step: "0.01", min: Some("0") } },
            Field { name: "tokens_per_hour", label: "Tokens / hour", hint: "Default 50,000", value: fmt_f(tph, 0),
                kind: FieldKind::Number { step: "1000", min: Some("0") } },
            Field { name: "model_per_million", label: "Model $ / M tok", hint: "Default 0.50", value: fmt_f(mpm, 2),
                kind: FieldKind::Number { step: "0.10", min: Some("0") } },
            Field { name: "max_hours", label: "Max hours to plot", hint: "Default 1000", value: fmt_f(max_h, 0),
                kind: FieldKind::Number { step: "100", min: Some("10") } },
        ],
    );

    let body = format!(
        r#"<h1>Harness pricing: OpenAI vs Anthropic</h1>
<p class="subtitle">At low usage, sandbox compute fees dominate. As session hours grow, model inference costs converge — both providers approach the same per-token economics.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Cost curves", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Sample hours", &tbl),
    );

    Html(page("Pricing", "/pricing", &body))
}

// =====================================================================
// Throughput vs perfection
// =====================================================================

#[derive(Deserialize)]
struct ThroughputParams {
    prs_per_day: Option<f64>,
    review_hours_per_pr: Option<f64>,
    num_reviewers: Option<usize>,
    error_rate: Option<f64>,
    test_pass_rate: Option<f64>,
    auto_fix_success: Option<f64>,
}

async fn throughput(Query(p): Query<ThroughputParams>) -> Html<String> {
    use physics::throughput::{
        net_fast_throughput, throughput_blocking, throughput_minimally_blocking, throughput_speedup,
    };

    let prs = p.prs_per_day.unwrap_or(100.0);
    let rh = p.review_hours_per_pr.unwrap_or(4.0);
    let nr = p.num_reviewers.unwrap_or(3).max(1);
    let er = p.error_rate.unwrap_or(0.05).clamp(0.0, 1.0);
    let tr = p.test_pass_rate.unwrap_or(0.95).clamp(0.0, 1.0);
    let af = p.auto_fix_success.unwrap_or(0.80).clamp(0.0, 1.0);

    let (b_clean, b_stuck) = throughput_blocking(prs, rh, nr, er);
    let (mb_merged, mb_fixed, mb_remaining) = throughput_minimally_blocking(prs, tr, er, af);
    let net_fast = net_fast_throughput(prs, tr, er, af);
    let speedup = throughput_speedup(prs, rh, nr, er, tr, af);

    let stats = stats_grid(&[
        Stat::new("Blocking: merged", fmt_f(b_clean, 1), format!("Stuck in queue: {}", fmt_f(b_stuck, 1))),
        Stat::new("Minimal: merged", fmt_f(mb_merged, 1), format!("Auto-fixed: {}, remaining: {}", fmt_f(mb_fixed, 1), fmt_f(mb_remaining, 1))),
        Stat::new("Net healthy/day", fmt_f(net_fast, 1), "merged − unfixed errors"),
        Stat::new("Speedup", format!("{:.1}×", speedup), "minimal vs blocking"),
    ]);

    let bars = vec![
        ("Blocking merged", b_clean, "#22c55e"),
        ("Blocking stuck", b_stuck, "#ef4444"),
        ("Minimal merged", mb_merged, "#00d4ff"),
        ("Auto-fixed", mb_fixed, "#a855f7"),
        ("Remaining errors", mb_remaining, "#f59e0b"),
    ];
    let chart = BarChart {
        title: "PRs per day by approach",
        y_label: "PRs / day",
        bars,
    }
    .render();

    let rows = vec![
        vec!["Total submitted".into(), fmt_f(prs, 1), "—".into()],
        vec!["Blocking merged".into(), fmt_f(b_clean, 1), format!("{:.1}%", 100.0 * b_clean / prs)],
        vec!["Blocking stuck in queue".into(), fmt_f(b_stuck, 1), format!("{:.1}%", 100.0 * b_stuck / prs)],
        vec!["Minimal: merged".into(), fmt_f(mb_merged, 1), format!("{:.1}%", 100.0 * mb_merged / prs)],
        vec!["Minimal: auto-fixed".into(), fmt_f(mb_fixed, 1), format!("{:.1}%", 100.0 * mb_fixed / prs)],
        vec!["Minimal: remaining errors".into(), fmt_f(mb_remaining, 1), format!("{:.1}%", 100.0 * mb_remaining / prs)],
    ];
    let tbl = table(&["Outcome", "PRs/day", "Share"], &rows);

    let form_html = form(
        "/throughput",
        &[
            Field { name: "prs_per_day", label: "PRs / day", hint: "Default 100", value: fmt_f(prs, 0),
                kind: FieldKind::Number { step: "10", min: Some("1") } },
            Field { name: "review_hours_per_pr", label: "Review hours / PR", hint: "Default 4", value: fmt_f(rh, 1),
                kind: FieldKind::Number { step: "0.5", min: Some("0.1") } },
            Field { name: "num_reviewers", label: "Reviewers", hint: "Default 3", value: nr.to_string(),
                kind: FieldKind::Number { step: "1", min: Some("1") } },
            Field { name: "error_rate", label: "Error rate", hint: "0–1, default 0.05", value: fmt_f(er, 3),
                kind: FieldKind::Number { step: "0.01", min: Some("0") } },
            Field { name: "test_pass_rate", label: "Test pass rate", hint: "0–1, default 0.95", value: fmt_f(tr, 3),
                kind: FieldKind::Number { step: "0.01", min: Some("0") } },
            Field { name: "auto_fix_success", label: "Auto-fix success", hint: "0–1, default 0.80", value: fmt_f(af, 3),
                kind: FieldKind::Number { step: "0.01", min: Some("0") } },
        ],
    );

    let body = format!(
        r#"<h1>Throughput vs perfection</h1>
<p class="subtitle">When agent PR output exceeds human review capacity, blocking review starves the pipeline. Minimally-blocking merge with auto-fix keeps net healthy throughput high.</p>
{f}{s}{c}{t}"#,
        f = section("Parameters", &form_html),
        s = stats,
        c = section("Outcome breakdown", &format!(r#"<div class="chart">{}</div>"#, chart)),
        t = section("Per-outcome details", &tbl),
    );

    Html(page("Throughput", "/throughput", &body))
}
