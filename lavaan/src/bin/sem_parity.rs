use std::env;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use lavaan::{Estimation, SemEngine, SemEngineImpl, SemInput};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,lavaan=trace"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();

    let args: Vec<String> = env::args().collect();
    let mut model_path = None;
    let mut s_path = None;
    let mut v_path = None;
    let mut wls_v_path = None;
    let mut nobs_path = None;
    let mut names = None;
    let mut estimation = Estimation::Dwls;
    let mut out_dir = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            "--s" => {
                i += 1;
                s_path = args.get(i).cloned();
            }
            "--v" => {
                i += 1;
                v_path = args.get(i).cloned();
            }
            "--wls-v" => {
                i += 1;
                wls_v_path = args.get(i).cloned();
            }
            "--nobs" => {
                i += 1;
                nobs_path = args.get(i).cloned();
            }
            "--names" => {
                i += 1;
                names = args.get(i).cloned();
            }
            "--estimation" => {
                i += 1;
                if let Some(est) = args.get(i) {
                    estimation = match est.as_str() {
                        "ML" | "Ml" | "ml" => Estimation::Ml,
                        _ => Estimation::Dwls,
                    };
                }
            }
            "--out" => {
                i += 1;
                out_dir = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.context("--model required")?;
    let s_path = s_path.context("--s required")?;
    let v_path = v_path.context("--v required")?;
    let names = names.context("--names required")?;

    let model = fs::read_to_string(&model_path).context("read model")?;
    let s = read_matrix(&s_path).context("read S")?;
    let v = read_matrix(&v_path).context("read V")?;
    let wls_v = if let Some(path) = wls_v_path {
        Some(read_matrix(&path).context("read WLS.V")?)
    } else {
        None
    };
    let n_obs = if let Some(path) = nobs_path {
        Some(read_scalar(&path).context("read nobs")?)
    } else {
        None
    };

    let names: Vec<String> = names
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let input = SemInput {
        s,
        v,
        wls_v,
        model,
        model_table: None,
        estimation,
        toler: None,
        std_lv: false,
        fix_measurement: false,
        q_snp: false,
        names,
        n_obs,
        optim_dx_tol: None,
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    };

    let engine = SemEngineImpl;
    let fit = engine.fit(&input).context("fit model")?;

    if let Some(out_dir) = out_dir {
        fs::create_dir_all(&out_dir).context("create output dir")?;
        write_params(Path::new(&out_dir).join("params.tsv"), &fit)?;
        write_defined(Path::new(&out_dir).join("defined.tsv"), &fit)?;
        write_stats(Path::new(&out_dir).join("stats.tsv"), &fit)?;
    } else {
        write_params("/dev/stdout", &fit)?;
    }

    Ok(())
}

fn read_matrix(path: &str) -> Result<Vec<Vec<f64>>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();
    let header = lines.next().context("matrix header")?;
    let dims: Vec<usize> = header
        .split_whitespace()
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse matrix dimensions")?;
    let (n, m) = if dims.len() == 1 {
        (dims[0], dims[0])
    } else if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        return Err(anyhow::anyhow!("matrix header must be 'n m'"));
    };

    let mut matrix = vec![vec![0.0; m]; n];
    for row in matrix.iter_mut().take(n) {
        let line = lines.next().context("matrix row")?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("parse matrix row")?;
        if vals.len() != m {
            return Err(anyhow::anyhow!("matrix row length mismatch"));
        }
        row[..m].copy_from_slice(&vals[..m]);
    }
    Ok(matrix)
}

fn read_scalar(path: &str) -> Result<f64> {
    let content = fs::read_to_string(path)?;
    let value = content
        .lines()
        .next()
        .context("scalar line")?
        .trim()
        .parse::<f64>()?;
    Ok(value)
}

fn write_params<P: AsRef<Path>>(path: P, fit: &lavaan::SemFit) -> Result<()> {
    let mut out = String::new();
    out.push_str("lhs\top\trhs\test\tse\tstd_all\n");
    for p in &fit.params {
        let std_all = p.est_std_all.unwrap_or(f64::NAN);
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\n",
            p.lhs, p.op, p.rhs, p.est, p.se, std_all
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_defined<P: AsRef<Path>>(path: P, fit: &lavaan::SemFit) -> Result<()> {
    let mut out = String::new();
    out.push_str("name\test\tse\texpr\n");
    for d in &fit.defined {
        out.push_str(&format!(
            "{}\t{:.10}\t{:.10}\t{}\n",
            d.name, d.est, d.se, d.expr
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_stats<P: AsRef<Path>>(path: P, fit: &lavaan::SemFit) -> Result<()> {
    let stats = &fit.stats;
    let out = format!(
        "chisq\tdf\taic\tcfi\tsrmr\tp_chisq\n{:.10}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\n",
        stats.chisq, stats.df, stats.aic, stats.cfi, stats.srmr, stats.p_chisq,
    );
    fs::write(path, out)?;
    Ok(())
}
