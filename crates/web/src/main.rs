//! Training Modern AI — Web Server

use std::env;

mod routes;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let port = env::var("PORT").unwrap_or_else(|_| "2718".into());
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let addr = format!("{host}:{port}");

    let app = routes::create_router();
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("Training Modern AI — Rust Edition");
    println!("  http://{addr}");
    println!("  API: /api/roofline, /api/cost, /api/context");
    println!("       /api/agents_md, /api/coordination, /api/health");

    axum::serve(listener, app).await?;
    Ok(())
}
