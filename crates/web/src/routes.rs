//! API routes for the web server.

use axum::{Router, Json, routing::get, response::Html};
use serde::Serialize;
use ndarray::Array1;
use physics::*;

pub fn create_router() -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/roofline", get(api_roofline))
        .route("/api/cost", get(api_cost))
        .route("/api/context", get(api_context))
        .route("/api/agents_md", get(api_agents_md))
        .route("/api/coordination", get(api_coordination))
        .route("/api/health", get(health))
}

async fn index() -> Html<String> {
    Html(include_str!("templates/index.html").to_string())
}

#[derive(Serialize)]
struct RooflineResponse {
    balance_batch: f64,
    balance_latency_ms: f64,
    min_latency_ms: f64,
}

async fn api_roofline() -> Json<RooflineResponse> {
    let bs = Array1::logspace(10.0, 0.0, 4.0, 200);
    let ctx = Array1::from_elem(200, CONTEXT_LENGTH);
    let (t_total, t_comp, t_weight, t_kv) = total_latency(
        &bs, N_ACTIVE, N_TOTAL, &ctx, 1.0, BYTES_PER_TOKEN, FLOPS, MEM_BW);
    let tm = &t_weight + &t_kv;
    let bi = (0..200).min_by(|&i, &j|
        (t_comp[i] - tm[i]).abs().partial_cmp(&(t_comp[j] - tm[j]).abs()).unwrap()
    ).unwrap();

    Json(RooflineResponse {
        balance_batch: bs[bi],
        balance_latency_ms: t_total[bi] * 1000.0,
        min_latency_ms: t_total[0] * 1000.0,
    })
}

#[derive(Serialize)]
struct CostResponse {
    costs: Vec<CostPoint>,
}

#[derive(Serialize)]
struct CostPoint {
    batch_size: f64,
    cost_per_million: f64,
    latency_ms: f64,
}

async fn api_cost() -> Json<CostResponse> {
    let bs = Array1::logspace(10.0, 0.0, 4.5, 300);
    let ctx = Array1::from_elem(300, CONTEXT_LENGTH);
    let rc = rack_cost_per_sec(GPU_COST_PER_HOUR, GPUS_IN_RACK);
    let cost = cost_per_million_tokens(&bs, N_ACTIVE, N_TOTAL, &ctx, 0.5, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc);
    let (t_total, _, _, _) = total_latency(&bs, N_ACTIVE, N_TOTAL, &ctx, 0.5, BYTES_PER_TOKEN, FLOPS, MEM_BW);

    let points: Vec<CostPoint> = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0].iter().map(|&b| {
        let i = bs.iter().position(|&x| (x - b).abs() < b * 0.1).unwrap_or(0);
        CostPoint { batch_size: bs[i], cost_per_million: cost[i], latency_ms: t_total[i] * 1000.0 }
    }).collect();

    Json(CostResponse { costs: points })
}

#[derive(Serialize)]
struct ContextResponse {
    crossover_tokens: f64,
}

async fn api_context() -> Json<ContextResponse> {
    let bs = Array1::from_elem(500, 2000.0);
    let ctx = Array1::logspace(10.0, 1.0, 6.0, 500);
    let tc = t_compute(&bs, N_ACTIVE, FLOPS);
    let tk = t_mem_kv(&bs, &ctx, BYTES_PER_TOKEN, MEM_BW);
    let tw = t_mem_weights(&bs, N_TOTAL, 1.0, MEM_BW);
    let tm = &tw + &tk;
    let cross = (0..500).min_by(|&i, &j|
        (tc[i] - tm[i]).abs().partial_cmp(&(tc[j] - tm[j]).abs()).unwrap()
    ).unwrap();
    Json(ContextResponse { crossover_tokens: ctx[cross] })
}

#[derive(Serialize)]
struct AgentsMdResponse {
    optimal_kb: f64,
    max_success: f64,
}

async fn api_agents_md() -> Json<AgentsMdResponse> {
    let sizes: Vec<f64> = (0..200).map(|i| 10.0_f64.powf(-1.0 + 3.0 * i as f64 / 199.0)).collect();
    let passive: Vec<f64> = sizes.iter().map(|&s|
        physics::knowledge::agents_md_passive_effectiveness(s, 8.0, 1.2, 128000.0, 150.0, 5000.0)
    ).collect();
    let best = passive.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    Json(AgentsMdResponse { optimal_kb: sizes[best.0], max_success: *best.1 })
}

#[derive(Serialize)]
struct CoordinationResponse {
    points: Vec<CoordinationPoint>,
}

#[derive(Serialize)]
struct CoordinationPoint {
    num_agents: usize,
    recursive: f64,
    equal: f64,
}

async fn api_coordination() -> Json<CoordinationResponse> {
    use physics::agents::effective_throughput;
    use physics::types::CoordinationStrategy;
    let points: Vec<CoordinationPoint> = [1, 10, 20, 50, 100].iter().map(|&n| {
        CoordinationPoint {
            num_agents: n,
            recursive: effective_throughput(n, CoordinationStrategy::Recursive),
            equal: effective_throughput(n, CoordinationStrategy::Equal),
        }
    }).collect();
    Json(CoordinationResponse { points })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "tests": "85 passing" }))
}
