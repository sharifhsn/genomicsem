use anyhow::{Context, Result};
use rayon::ThreadPoolBuilder;

pub fn run_in_pool<T, F>(cores: Option<usize>, context: &'static str, f: F) -> Result<T>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    if let Some(cores) = cores {
        let pool = ThreadPoolBuilder::new()
            .num_threads(cores)
            .build()
            .context(context)?;
        Ok(pool.install(f))
    } else {
        Ok(f())
    }
}

pub fn collect_results<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(results.len());
    for res in results {
        out.push(res?);
    }
    Ok(out)
}

pub fn resolve_threads(cores: Option<usize>, tasks: usize) -> Option<usize> {
    if let Some(cores) = cores {
        let capped = cores.min(tasks.max(1));
        if cores > capped {
            tracing::warn!(
                "Provided cores ({cores}) greater than number of tasks ({tasks}); using {capped}"
            );
        }
        Some(capped)
    } else {
        None
    }
}
