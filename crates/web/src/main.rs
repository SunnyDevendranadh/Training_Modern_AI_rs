//! Training Modern AI — Axum web server entry point.

use std::env;

mod chart;
mod render;
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
    println!("  Listening on http://{addr}");
    println!("  Open the URL in a browser. All pages are server-rendered HTML.");

    axum::serve(listener, app).await?;
    Ok(())
}
