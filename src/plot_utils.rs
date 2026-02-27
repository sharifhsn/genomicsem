use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub fn ensure_plots_dir() -> Result<PathBuf> {
    let dir = PathBuf::from("Plots");
    fs::create_dir_all(&dir).context("create Plots directory")?;
    Ok(dir)
}

pub fn plot_path(prefix: Option<&str>, name: &str) -> PathBuf {
    let file_name = match prefix {
        Some(pfx) => format!("{pfx}_{name}.html"),
        None => format!("{name}.html"),
    };
    PathBuf::from("Plots").join(file_name)
}
