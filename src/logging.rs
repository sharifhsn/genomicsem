use std::fs::File;
use std::io::Write;

use anyhow::Result;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

pub fn log_line(log: &mut File, message: &str, print: bool) -> Result<()> {
    if print {
        info!("{message}");
    }
    writeln!(log, "{message}")?;
    Ok(())
}

pub fn warn_line(log: &mut File, message: &str) -> Result<()> {
    warn!("{message}");
    writeln!(log, "{message}")?;
    Ok(())
}
