//! Web server: Axum-based, fully server-rendered HTML for the training app.
//!
//! Every page is computed on the server and returned as a complete HTML
//! document with embedded SVG charts. The client does not need JavaScript.

pub mod chart;
pub mod render;
pub mod routes;
