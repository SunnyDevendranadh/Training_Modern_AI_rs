//! Placeholder utilities for experiments.
//! For full chart generation, see the Python `experiments/` directory.

use std::path::Path;
use std::io::Write;

/// Write a summary text file instead of a PNG chart.
/// The Rust crate focuses on computation; visualization remains in Python.
pub fn write_summary(path: &Path, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{}", content)?;
    Ok(())
}
