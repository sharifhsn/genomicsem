use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use nlopt::{Algorithm, Nlopt, Target, approximate_gradient};
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use tracing::{info, warn};

use crate::io::read_table;
use crate::types::{LdscOutput, Matrix};

#[derive(Debug, Clone)]
pub enum HdlMethod {
    Piecewise,
    Jackknife,
}

#[derive(Debug, Clone)]
pub struct HdlConfig {
    pub traits: Vec<PathBuf>,
    pub sample_prev: Vec<Option<f64>>,
    pub population_prev: Vec<Option<f64>>,
    pub ld_path: PathBuf,
    pub n_ref: usize,
    pub trait_names: Option<Vec<String>>,
    pub method: HdlMethod,
}

pub fn hdl(config: &HdlConfig) -> Result<LdscOutput> {
    if has_rda_reference(&config.ld_path)? {
        // R HDL expects .rda + UKB_snp_counter files. Parsing .rda requires
        // an external dependency not currently included, so we hard-fail here.
        todo!("HDL .rda reference panels are not yet supported in Rust");
    }

    let pieces = scan_ld_pieces(&config.ld_path)?;
    if pieces.is_empty() {
        return Err(anyhow::anyhow!(
            "No LD pieces found in {}",
            config.ld_path.display()
        ));
    }

    let n_traits = config.traits.len();
    if n_traits < 2 {
        warn!("HDL requires 2 or more traits.");
    }
    let trait_names = resolve_trait_names(config, n_traits)?;

    let gwas = load_gwas(&config.traits)?;
    let n_v = n_traits * (n_traits + 1) / 2;
    let num_pieces = pieces.len();

    let mut s = vec![vec![f64::NAN; n_traits]; n_traits];
    let mut i_mat = vec![vec![f64::NAN; n_traits]; n_traits];
    let mut v_hold = vec![vec![0.0; n_v]; num_pieces];
    let mut n_vec = vec![f64::NAN; n_v];

    let mut idx = 0usize;
    for j in 0..n_traits {
        for d in j..n_traits {
            info!("Estimating HDL cell {} of {}", idx + 1, n_v);
            if j == d {
                match config.method {
                    HdlMethod::Piecewise => {
                        let hdl = run_hdl_diag_piecewise(&gwas[j], &pieces, config.n_ref as f64)?;
                        let h2_sum: f64 = hdl.h2.iter().sum();
                        s[j][j] = h2_sum;
                        i_mat[j][j] = mean(&hdl.intercept);
                        for (p, row) in v_hold.iter_mut().enumerate().take(num_pieces) {
                            row[idx] = jackknife_piecewise(&hdl.h2, p);
                        }
                        n_vec[idx] = gwas[j].n_median;
                    }
                    HdlMethod::Jackknife => {
                        let hdl = run_hdl_diag_jackknife(&gwas[j], &pieces, config.n_ref as f64)?;
                        s[j][j] = hdl.h2;
                        i_mat[j][j] = hdl.intercept;
                        for (p, row) in v_hold.iter_mut().enumerate().take(num_pieces) {
                            row[idx] = hdl.jackknife[p];
                        }
                        n_vec[idx] = gwas[j].n_median;
                    }
                }
            } else {
                match config.method {
                    HdlMethod::Piecewise => {
                        let hdl = run_hdl_offdiag_piecewise(
                            &gwas[j],
                            &gwas[d],
                            &pieces,
                            config.n_ref as f64,
                        )?;
                        let cov_sum: f64 = hdl.h12.iter().sum();
                        s[j][d] = cov_sum;
                        s[d][j] = cov_sum;
                        i_mat[j][d] = mean(&hdl.intercept);
                        i_mat[d][j] = i_mat[j][d];
                        for (p, row) in v_hold.iter_mut().enumerate().take(num_pieces) {
                            row[idx] = jackknife_piecewise(&hdl.h12, p);
                        }
                        n_vec[idx] = (gwas[j].n_median * gwas[d].n_median).sqrt();
                    }
                    HdlMethod::Jackknife => {
                        let hdl = run_hdl_offdiag_jackknife(
                            &gwas[j],
                            &gwas[d],
                            &pieces,
                            config.n_ref as f64,
                        )?;
                        s[j][d] = hdl.h12;
                        s[d][j] = hdl.h12;
                        i_mat[j][d] = hdl.intercept;
                        i_mat[d][j] = i_mat[j][d];
                        for (p, row) in v_hold.iter_mut().enumerate().take(num_pieces) {
                            row[idx] = hdl.jackknife[p];
                        }
                        n_vec[idx] = (gwas[j].n_median * gwas[d].n_median).sqrt();
                    }
                }
            }
            idx += 1;
        }
    }

    let mut v = covariance(&v_hold);
    let scale = (num_pieces as f64 - 1.0).max(0.0);
    for row in &mut v {
        for val in row {
            *val *= scale;
        }
    }

    let liab = liability_vector(&config.sample_prev, &config.population_prev)?;
    let s_scaled = scale_liability(&s, &liab);
    let scale_o = lower_triangle_ratio(&s_scaled, &s);
    let v_scaled = scale_v(&v, &scale_o);

    let m_ref = pieces.iter().map(|p| p.ldsc.len()).sum::<usize>();

    Ok(LdscOutput {
        s: s_scaled,
        v: v_scaled,
        i: i_mat,
        n: n_vec,
        m: m_ref,
        s_stand: None,
        v_stand: None,
        trait_names,
    })
}

fn has_rda_reference(path: &Path) -> Result<bool> {
    let mut has_counter = false;
    let mut has_rda = false;
    for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.contains("UKB_snp_counter") {
            has_counter = true;
        }
        if name.ends_with(".rda") {
            has_rda = true;
        }
    }
    Ok(has_counter && has_rda)
}

struct GwasData {
    n_median: f64,
    snp_index: HashMap<String, usize>,
    z: Vec<f64>,
    n: Vec<f64>,
    a2: Vec<String>,
}

struct LdPiece {
    ldsc: Vec<f64>,
    lam: Vec<f64>,
    v: Matrix,
    snps: Vec<String>,
    a2: Vec<String>,
}

struct HdlDiagResult {
    h2: Vec<f64>,
    intercept: Vec<f64>,
}

struct HdlOffResult {
    h12: Vec<f64>,
    intercept: Vec<f64>,
}

struct HdlDiagJackknife {
    h2: f64,
    intercept: f64,
    jackknife: Vec<f64>,
}

struct HdlOffJackknife {
    h12: f64,
    intercept: f64,
    jackknife: Vec<f64>,
}

fn run_hdl_diag_piecewise(
    gwas: &GwasData,
    pieces: &[LdPiece],
    n_ref: f64,
) -> Result<HdlDiagResult> {
    let mut h2 = Vec::with_capacity(pieces.len());
    let mut intercept = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let bhat = build_bhat(gwas, piece);
        let a11 = bhat.iter().map(|v| v * v).collect::<Vec<_>>();
        let (h2_wls, _) = wls_h2(&a11, &piece.ldsc, gwas.n_median)?;
        let bstar = mat_t_vec(&piece.v, &bhat);
        let params = optimize_llfun(
            vec![h2_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar,
            piece.ldsc.len() as f64,
            gwas.n_median,
            n_ref,
        )?;
        h2.push(params[0]);
        intercept.push(params[1]);
    }
    Ok(HdlDiagResult { h2, intercept })
}

fn run_hdl_offdiag_piecewise(
    g1: &GwasData,
    g2: &GwasData,
    pieces: &[LdPiece],
    n_ref: f64,
) -> Result<HdlOffResult> {
    let rho12 = gwas_corr(g1, g2);
    let n0 =
        g1.n.iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(g2.n.iter().cloned().fold(f64::INFINITY, f64::min));
    let p1 = if g1.n_median != 0.0 {
        n0 / g1.n_median
    } else {
        0.0
    };
    let p2 = if g2.n_median != 0.0 {
        n0 / g2.n_median
    } else {
        0.0
    };

    let mut h12 = Vec::with_capacity(pieces.len());
    let mut intercept = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let bhat1 = build_bhat(g1, piece);
        let bhat2 = build_bhat(g2, piece);
        let a11 = bhat1.iter().map(|v| v * v).collect::<Vec<_>>();
        let a22 = bhat2.iter().map(|v| v * v).collect::<Vec<_>>();
        let a12 = bhat1
            .iter()
            .zip(&bhat2)
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        let (h11_wls, _h11_ols) = wls_h2(&a11, &piece.ldsc, g1.n_median)?;
        let (h22_wls, _h22_ols) = wls_h2(&a22, &piece.ldsc, g2.n_median)?;
        let (h12_wls, _h12_ols) = wls_h2(&a12, &piece.ldsc, (g1.n_median * g2.n_median).sqrt())?;

        let bstar1 = mat_t_vec(&piece.v, &bhat1);
        let bstar2 = mat_t_vec(&piece.v, &bhat2);

        let h11 = optimize_llfun(
            vec![h11_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar1,
            piece.ldsc.len() as f64,
            g1.n_median,
            n_ref,
        )?;

        let h22 = optimize_llfun(
            vec![h22_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar2,
            piece.ldsc.len() as f64,
            g2.n_median,
            n_ref,
        )?;

        let h12_params = optimize_llfun_gcov(
            vec![h12_wls, rho12],
            [-1.0, -10.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar1,
            &bstar2,
            piece.ldsc.len() as f64,
            g1.n_median,
            g2.n_median,
            n0,
            n_ref,
            p1,
            p2,
            h11,
            h22,
        )?;

        h12.push(h12_params[0]);
        intercept.push(h12_params[1]);
    }

    Ok(HdlOffResult { h12, intercept })
}

fn run_hdl_diag_jackknife(
    gwas: &GwasData,
    pieces: &[LdPiece],
    n_ref: f64,
) -> Result<HdlDiagJackknife> {
    let mut per_piece = Vec::with_capacity(pieces.len());
    let mut bstar_v = Vec::with_capacity(pieces.len());
    let mut lam_v = Vec::with_capacity(pieces.len());

    for piece in pieces {
        let bhat = build_bhat(gwas, piece);
        let a11 = bhat.iter().map(|v| v * v).collect::<Vec<_>>();
        let (h2_wls, _) = wls_h2(&a11, &piece.ldsc, gwas.n_median)?;
        let bstar = mat_t_vec(&piece.v, &bhat);
        let params = optimize_llfun(
            vec![h2_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar,
            piece.ldsc.len() as f64,
            gwas.n_median,
            n_ref,
        )?;
        per_piece.push(params);
        bstar_v.push(bstar);
        lam_v.push(piece.lam.clone());
    }

    let m_ref: f64 = pieces.iter().map(|p| p.ldsc.len() as f64).sum();
    let start = vec![per_piece.iter().map(|p| p[0]).sum::<f64>(), 1.0];
    let full = optimize_llfun(
        start,
        [0.0, 0.0],
        [1.0, 10.0],
        &concat_excluding(&lam_v, None),
        &concat_excluding(&bstar_v, None),
        m_ref,
        gwas.n_median,
        n_ref,
    )?;

    let jackknife = (0..pieces.len())
        .into_par_iter()
        .map(|i| {
            let lam = concat_excluding(&lam_v, Some(i));
            let bstar = concat_excluding(&bstar_v, Some(i));
            optimize_llfun(
                full.clone(),
                [0.0, 0.0],
                [1.0, 10.0],
                &lam,
                &bstar,
                m_ref,
                gwas.n_median,
                n_ref,
            )
            .map(|v| v[0])
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(HdlDiagJackknife {
        h2: full[0],
        intercept: full[1],
        jackknife,
    })
}

fn run_hdl_offdiag_jackknife(
    g1: &GwasData,
    g2: &GwasData,
    pieces: &[LdPiece],
    n_ref: f64,
) -> Result<HdlOffJackknife> {
    let rho12 = gwas_corr(g1, g2);
    let n0 =
        g1.n.iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(g2.n.iter().cloned().fold(f64::INFINITY, f64::min));
    let p1 = if g1.n_median != 0.0 {
        n0 / g1.n_median
    } else {
        0.0
    };
    let p2 = if g2.n_median != 0.0 {
        n0 / g2.n_median
    } else {
        0.0
    };

    let mut per_piece = Vec::with_capacity(pieces.len());
    let mut bstar1_v = Vec::with_capacity(pieces.len());
    let mut bstar2_v = Vec::with_capacity(pieces.len());
    let mut lam_v = Vec::with_capacity(pieces.len());

    for piece in pieces {
        let bhat1 = build_bhat(g1, piece);
        let bhat2 = build_bhat(g2, piece);
        let a11 = bhat1.iter().map(|v| v * v).collect::<Vec<_>>();
        let a22 = bhat2.iter().map(|v| v * v).collect::<Vec<_>>();
        let a12 = bhat1
            .iter()
            .zip(&bhat2)
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        let (h11_wls, _h11_ols) = wls_h2(&a11, &piece.ldsc, g1.n_median)?;
        let (h22_wls, _h22_ols) = wls_h2(&a22, &piece.ldsc, g2.n_median)?;
        let (h12_wls, _h12_ols) = wls_h2(&a12, &piece.ldsc, (g1.n_median * g2.n_median).sqrt())?;

        let bstar1 = mat_t_vec(&piece.v, &bhat1);
        let bstar2 = mat_t_vec(&piece.v, &bhat2);

        let h11 = optimize_llfun(
            vec![h11_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar1,
            piece.ldsc.len() as f64,
            g1.n_median,
            n_ref,
        )?;

        let h22 = optimize_llfun(
            vec![h22_wls, 1.0],
            [0.0, 0.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar2,
            piece.ldsc.len() as f64,
            g2.n_median,
            n_ref,
        )?;

        let h12_params = optimize_llfun_gcov(
            vec![h12_wls, rho12],
            [-1.0, -10.0],
            [1.0, 10.0],
            &piece.lam,
            &bstar1,
            &bstar2,
            piece.ldsc.len() as f64,
            g1.n_median,
            g2.n_median,
            n0,
            n_ref,
            p1,
            p2,
            h11.clone(),
            h22.clone(),
        )?;

        per_piece.push((h11, h22, h12_params));
        bstar1_v.push(bstar1);
        bstar2_v.push(bstar2);
        lam_v.push(piece.lam.clone());
    }

    let m_ref: f64 = pieces.iter().map(|p| p.ldsc.len() as f64).sum();
    let h11_sum: f64 = per_piece.iter().map(|p| p.0[0]).sum();
    let h22_sum: f64 = per_piece.iter().map(|p| p.1[0]).sum();
    let h12_sum: f64 = per_piece.iter().map(|p| p.2[0]).sum();

    let full_h11 = optimize_llfun(
        vec![h11_sum, 1.0],
        [0.0, 0.0],
        [1.0, 10.0],
        &concat_excluding(&lam_v, None),
        &concat_excluding(&bstar1_v, None),
        m_ref,
        g1.n_median,
        n_ref,
    )?;
    let full_h22 = optimize_llfun(
        vec![h22_sum, 1.0],
        [0.0, 0.0],
        [1.0, 10.0],
        &concat_excluding(&lam_v, None),
        &concat_excluding(&bstar2_v, None),
        m_ref,
        g2.n_median,
        n_ref,
    )?;

    let full_h12 = optimize_llfun_gcov_with_fallback(
        vec![h12_sum, rho12],
        &concat_excluding(&lam_v, None),
        &concat_excluding(&bstar1_v, None),
        &concat_excluding(&bstar2_v, None),
        m_ref,
        g1.n_median,
        g2.n_median,
        n0,
        n_ref,
        p1,
        p2,
        full_h11.clone(),
        full_h22.clone(),
    )?;

    let jackknife = (0..pieces.len())
        .into_par_iter()
        .map(|i| {
            let lam = concat_excluding(&lam_v, Some(i));
            let b1 = concat_excluding(&bstar1_v, Some(i));
            let b2 = concat_excluding(&bstar2_v, Some(i));
            optimize_llfun_gcov(
                full_h12.clone(),
                [-1.0, -10.0],
                [1.0, 10.0],
                &lam,
                &b1,
                &b2,
                m_ref,
                g1.n_median,
                g2.n_median,
                n0,
                n_ref,
                p1,
                p2,
                full_h11.clone(),
                full_h22.clone(),
            )
            .map(|v| v[0])
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(HdlOffJackknife {
        h12: full_h12[0],
        intercept: full_h12[1],
        jackknife,
    })
}

fn load_gwas(traits: &[PathBuf]) -> Result<Vec<GwasData>> {
    let mut out = Vec::with_capacity(traits.len());
    for path in traits {
        let mut df = read_table(path)?;
        ensure_column(&df, "SNP")?;
        ensure_column(&df, "N")?;
        ensure_column(&df, "A2")?;

        if df.column("Z").is_err() {
            ensure_column(&df, "b")?;
            ensure_column(&df, "se")?;
            let b = df.column("b")?.f64().context("b")?;
            let se = df.column("se")?.f64().context("se")?;
            let z: Vec<f64> = b
                .into_iter()
                .zip(se)
                .map(|(b, s)| match (b, s) {
                    (Some(b), Some(s)) if s != 0.0 => b / s,
                    _ => f64::NAN,
                })
                .collect();
            let z_series = Series::new("Z".into(), z);
            df.with_column(z_series.into())?;
        }

        let z = df.column("Z")?.f64().context("Z")?;
        let n = df.column("N")?.f64().context("N")?;
        let a2 = df.column("A2")?.str().context("A2")?;
        let snp = df.column("SNP")?.str().context("SNP")?;

        let mut snp_index = HashMap::new();
        let mut z_vec = Vec::with_capacity(df.height());
        let mut n_vec = Vec::with_capacity(df.height());
        let mut a2_vec = Vec::with_capacity(df.height());

        for idx in 0..df.height() {
            let snp_name = snp.get(idx).unwrap_or("").to_string();
            snp_index.insert(snp_name.clone(), idx);
            z_vec.push(z.get(idx).unwrap_or(f64::NAN));
            n_vec.push(n.get(idx).unwrap_or(f64::NAN));
            a2_vec.push(a2.get(idx).unwrap_or("").to_string());
        }

        let n_median = median(&n_vec);
        out.push(GwasData {
            n_median,
            snp_index,
            z: z_vec,
            n: n_vec,
            a2: a2_vec,
        });
    }
    Ok(out)
}

fn scan_ld_pieces(ld_path: &Path) -> Result<Vec<LdPiece>> {
    let mut pieces = Vec::new();
    for entry in fs::read_dir(ld_path).with_context(|| format!("read {}", ld_path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("bim") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let ldsc = find_with_suffix(ld_path, stem, ".ldsc")?;
        let lam = find_with_suffix(ld_path, stem, ".lam")?;
        let v = find_with_suffix(ld_path, stem, ".v")?;

        let (snps, a2) = read_bim(&path)?;
        let ldsc = read_vector(&ldsc)?;
        let lam = read_vector(&lam)?;
        let v = read_matrix(&v)?;

        if ldsc.len() != snps.len() || lam.len() != snps.len() {
            return Err(anyhow::anyhow!(
                "LD vectors length mismatch with BIM in {}",
                stem
            ));
        }
        if v.len() != snps.len() {
            return Err(anyhow::anyhow!(
                "V matrix rows mismatch with BIM in {}",
                stem
            ));
        }

        pieces.push(LdPiece {
            ldsc,
            lam,
            v,
            snps,
            a2,
        });
    }
    Ok(pieces)
}

fn read_bim(path: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let content = std::fs::read_to_string(path)?;
    let mut snps = Vec::new();
    let mut a2 = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }
        snps.push(parts[1].to_string());
        a2.push(parts[5].to_string());
    }
    Ok((snps, a2))
}

fn read_vector(path: &Path) -> Result<Vec<f64>> {
    let df = read_table(path)?;
    let first = df
        .get_column_names()
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("empty table {}", path.display()))?;
    let series = df.column(first)?.f64().context("vector")?;
    Ok(series.into_iter().map(|v| v.unwrap_or(0.0)).collect())
}

fn read_matrix(path: &Path) -> Result<Matrix> {
    let df = read_table(path)?;
    let mut out = Vec::with_capacity(df.height());
    for row in 0..df.height() {
        let mut vals = Vec::with_capacity(df.width());
        for name in df.get_column_names() {
            let series = df.column(name)?.f64().context("matrix")?;
            vals.push(series.get(row).unwrap_or(0.0));
        }
        out.push(vals);
    }
    Ok(out)
}

fn find_with_suffix(dir: &Path, stem: &str, suffix: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with(stem) && name.contains(suffix) {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Missing {suffix} for {stem}"))
}

fn build_bhat(gwas: &GwasData, piece: &LdPiece) -> Vec<f64> {
    let mut bhat = vec![0.0; piece.snps.len()];
    for (idx, snp) in piece.snps.iter().enumerate() {
        if let Some(&gidx) = gwas.snp_index.get(snp) {
            let z = gwas.z[gidx];
            let n = gwas.n[gidx];
            if n > 0.0 && z.is_finite() {
                let mut val = z / n.sqrt();
                if gwas.a2[gidx] != piece.a2[idx] {
                    val = -val;
                }
                bhat[idx] = val;
            }
        }
    }
    bhat
}

fn wls_h2(a: &[f64], ldsc: &[f64], n: f64) -> Result<(f64, f64)> {
    let (_intercept, slope) = ols(a, ldsc)?;
    let h2_ols = slope * ldsc.len() as f64;
    let var = variance_weights(ldsc, h2_ols, n);
    let weights: Vec<f64> = var
        .iter()
        .map(|v| if *v != 0.0 { 1.0 / v } else { 0.0 })
        .collect();
    let (_intercept_w, slope_w) = wls(a, ldsc, &weights)?;
    let h2_wls = slope_w * ldsc.len() as f64;
    Ok((h2_wls, h2_ols))
}

fn variance_weights(ldsc: &[f64], h2: f64, n: f64) -> Vec<f64> {
    let m = ldsc.len() as f64;
    ldsc.iter()
        .map(|l| {
            let val = h2 * l / m + 1.0 / n;
            val * val
        })
        .collect()
}

fn ols(y: &[f64], x: &[f64]) -> Result<(f64, f64)> {
    let n = y.len() as f64;
    let sx = x.iter().sum::<f64>();
    let sy = y.iter().sum::<f64>();
    let sxx = x.iter().map(|v| v * v).sum::<f64>();
    let sxy = x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let denom = n * sxx - sx * sx;
    if denom == 0.0 {
        return Err(anyhow::anyhow!("OLS denominator is zero"));
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    Ok((intercept, slope))
}

fn wls(y: &[f64], x: &[f64], w: &[f64]) -> Result<(f64, f64)> {
    let sw = w.iter().sum::<f64>();
    let swx = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>();
    let swy = y.iter().zip(w).map(|(y, w)| y * w).sum::<f64>();
    let swxx = x.iter().zip(w).map(|(x, w)| x * x * w).sum::<f64>();
    let swxy = x
        .iter()
        .zip(y)
        .zip(w)
        .map(|((x, y), w)| x * y * w)
        .sum::<f64>();
    let denom = sw * swxx - swx * swx;
    if denom == 0.0 {
        return Err(anyhow::anyhow!("WLS denominator is zero"));
    }
    let slope = (sw * swxy - swx * swy) / denom;
    let intercept = (swy - slope * swx) / sw;
    Ok((intercept, slope))
}

fn mat_t_vec(v: &Matrix, x: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += v[j][i] * x[j];
        }
        out[i] = sum;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn optimize_llfun(
    start: Vec<f64>,
    lower: [f64; 2],
    upper: [f64; 2],
    lam: &[f64],
    bstar: &[f64],
    m: f64,
    n: f64,
    n_ref: f64,
) -> Result<Vec<f64>> {
    let data = LlData {
        n,
        m,
        n_ref,
        lam: lam.to_vec(),
        bstar: bstar.to_vec(),
        lim: (-18.0f64).exp(),
    };
    let obj = |x: &[f64], grad: Option<&mut [f64]>, data: &mut LlData| -> f64 {
        let f = llfun_value(x, data);
        if let Some(g) = grad {
            approximate_gradient(x, |x| llfun_value(x, data), g);
        }
        f
    };
    let mut opt = Nlopt::new(Algorithm::Lbfgs, 2, obj, Target::Minimize, data);
    let _ = opt.set_lower_bounds(&lower);
    let _ = opt.set_upper_bounds(&upper);
    let _ = opt.set_ftol_rel(1e-7);
    let _ = opt.set_maxeval(1000);

    let mut x = start.clone();
    match opt.optimize(&mut x) {
        Ok(_) => Ok(x),
        Err((fail, _)) => {
            warn!("HDL optimizer failed: {:?}; returning start values", fail);
            Ok(start)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_llfun_gcov(
    start: Vec<f64>,
    lower: [f64; 2],
    upper: [f64; 2],
    lam: &[f64],
    bstar1: &[f64],
    bstar2: &[f64],
    m: f64,
    n1: f64,
    n2: f64,
    n0: f64,
    n_ref: f64,
    p1: f64,
    p2: f64,
    h11: Vec<f64>,
    h22: Vec<f64>,
) -> Result<Vec<f64>> {
    let data = LlGcovData {
        h11,
        h22,
        m,
        n1,
        n2,
        n0,
        n_ref,
        p1,
        p2,
        lam: lam.to_vec(),
        bstar1: bstar1.to_vec(),
        bstar2: bstar2.to_vec(),
        lim: (-18.0f64).exp(),
    };
    let obj = |x: &[f64], grad: Option<&mut [f64]>, data: &mut LlGcovData| -> f64 {
        let f = llfun_gcov_value(x, data);
        if let Some(g) = grad {
            approximate_gradient(x, |x| llfun_gcov_value(x, data), g);
        }
        f
    };
    let mut opt = Nlopt::new(Algorithm::Lbfgs, 2, obj, Target::Minimize, data);
    let _ = opt.set_lower_bounds(&lower);
    let _ = opt.set_upper_bounds(&upper);
    let _ = opt.set_ftol_rel(1e-7);
    let _ = opt.set_maxeval(1000);

    let mut x = start.clone();
    match opt.optimize(&mut x) {
        Ok(_) => Ok(x),
        Err((fail, _)) => {
            warn!(
                "HDL gcov optimizer failed: {:?}; returning start values",
                fail
            );
            Ok(start)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_llfun_gcov_with_fallback(
    start: Vec<f64>,
    lam: &[f64],
    bstar1: &[f64],
    bstar2: &[f64],
    m: f64,
    n1: f64,
    n2: f64,
    n0: f64,
    n_ref: f64,
    p1: f64,
    p2: f64,
    h11: Vec<f64>,
    h22: Vec<f64>,
) -> Result<Vec<f64>> {
    let mut out = optimize_llfun_gcov(
        start.clone(),
        [-1.0, -10.0],
        [1.0, 10.0],
        lam,
        bstar1,
        bstar2,
        m,
        n1,
        n2,
        n0,
        n_ref,
        p1,
        p2,
        h11.clone(),
        h22.clone(),
    )?;
    if out == start {
        let base = (h11[0] * h22[0]).max(0.0).sqrt();
        let candidates = [0.0, -0.5 * base, 0.5 * base];
        for c in candidates {
            let try_start = vec![c, start[1]];
            let candidate = optimize_llfun_gcov(
                try_start.clone(),
                [-1.0, -10.0],
                [1.0, 10.0],
                lam,
                bstar1,
                bstar2,
                m,
                n1,
                n2,
                n0,
                n_ref,
                p1,
                p2,
                h11.clone(),
                h22.clone(),
            )?;
            if candidate != try_start {
                out = candidate;
                break;
            }
        }
    }
    Ok(out)
}

struct LlData {
    n: f64,
    m: f64,
    n_ref: f64,
    lam: Vec<f64>,
    bstar: Vec<f64>,
    lim: f64,
}

fn llfun_value(param: &[f64], data: &LlData) -> f64 {
    if param.len() < 2 {
        return f64::INFINITY;
    }
    let h2 = param[0];
    let intercept = param[1];
    let mut ll = 0.0;
    for (lam, b) in data.lam.iter().zip(&data.bstar) {
        let mut lamh2 = h2 / data.m * lam * lam - h2 * lam / data.n_ref + intercept * lam / data.n;
        if lamh2 < data.lim {
            lamh2 = data.lim;
        }
        ll += lamh2.ln() + (b * b) / lamh2;
    }
    ll
}

struct LlGcovData {
    h11: Vec<f64>,
    h22: Vec<f64>,
    m: f64,
    n1: f64,
    n2: f64,
    n0: f64,
    n_ref: f64,
    p1: f64,
    p2: f64,
    lam: Vec<f64>,
    bstar1: Vec<f64>,
    bstar2: Vec<f64>,
    lim: f64,
}

fn llfun_gcov_value(param: &[f64], data: &LlGcovData) -> f64 {
    if param.len() < 2 {
        return f64::INFINITY;
    }
    let h12 = param[0];
    let intercept = param[1];
    let mut ll = 0.0;
    for i in 0..data.lam.len() {
        let lam = data.lam[i];
        let mut lam11 = data.h11[0] / data.m * lam * lam - data.h11[0] * lam / data.n_ref
            + data.h11[1] * lam / data.n1;
        let mut lam22 = data.h22[0] / data.m * lam * lam - data.h22[0] * lam / data.n_ref
            + data.h22[1] * lam / data.n2;
        if lam11 < data.lim {
            lam11 = data.lim;
        }
        if lam22 < data.lim {
            lam22 = data.lim;
        }

        let lam12 = if data.n0 > 0.0 {
            h12 / data.m * lam * lam + data.p1 * data.p2 * intercept * lam / data.n0
        } else {
            h12 / data.m * lam * lam
        };

        let ustar = data.bstar2[i] - lam12 / lam11 * data.bstar1[i];
        let mut lam22_1 = lam22 - lam12 * lam12 / lam11;
        if lam22_1 < data.lim {
            lam22_1 = data.lim;
        }
        ll += lam22_1.ln() + (ustar * ustar) / lam22_1;
    }
    ll
}

fn concat_excluding(values: &[Vec<f64>], skip: Option<usize>) -> Vec<f64> {
    let mut out = Vec::new();
    for (idx, v) in values.iter().enumerate() {
        if skip.is_some() && skip == Some(idx) {
            continue;
        }
        out.extend_from_slice(v);
    }
    out
}

fn gwas_corr(g1: &GwasData, g2: &GwasData) -> f64 {
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut sum1 = 0.0;
    let mut sum22 = 0.0;
    let mut n = 0.0;
    for (snp, &idx1) in &g1.snp_index {
        if let Some(&idx2) = g2.snp_index.get(snp) {
            let z1 = g1.z[idx1];
            let z2 = g2.z[idx2];
            if z1.is_finite() && z2.is_finite() {
                sum += z1 * z2;
                sum1 += z1;
                sum2 += z2;
                sum22 += z1 * z1;
                n += 1.0;
            }
        }
    }
    if n == 0.0 {
        return 0.0;
    }
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;
    let cov = sum / n - mean1 * mean2;
    let var1 = sum22 / n - mean1 * mean1;
    if var1 <= 0.0 {
        return 0.0;
    }
    cov / var1
}

fn jackknife_piecewise(values: &[f64], idx: usize) -> f64 {
    let n = values.len();
    if n <= 1 {
        return values.first().copied().unwrap_or(f64::NAN);
    }
    let sum: f64 = values.iter().sum();
    let leave_out = sum - values[idx];
    (n as f64 / (n as f64 - 1.0)) * leave_out
}

#[allow(clippy::needless_range_loop)]
fn covariance(matrix: &[Vec<f64>]) -> Matrix {
    if matrix.is_empty() {
        return vec![];
    }
    let n = matrix.len();
    let p = matrix[0].len();
    let mut means = vec![0.0; p];
    for row in matrix {
        for i in 0..p {
            means[i] += row[i];
        }
    }
    for i in 0..p {
        means[i] /= n as f64;
    }
    let mut cov = vec![vec![0.0; p]; p];
    for row in matrix {
        for i in 0..p {
            let di = row[i] - means[i];
            for j in 0..p {
                cov[i][j] += di * (row[j] - means[j]);
            }
        }
    }
    let denom = (n.saturating_sub(1)) as f64;
    if denom > 0.0 {
        for i in 0..p {
            for j in 0..p {
                cov[i][j] /= denom;
            }
        }
    }
    cov
}

fn liability_vector(sample_prev: &[Option<f64>], pop_prev: &[Option<f64>]) -> Result<Vec<f64>> {
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let mut out = vec![1.0; sample_prev.len()];
    for i in 0..sample_prev.len() {
        if let (Some(pop), Some(samp)) = (pop_prev[i], sample_prev[i]) {
            let t = normal.inverse_cdf(1.0 - pop);
            let z = normal.pdf(t);
            out[i] = (pop * (1.0 - pop)).powi(2) / (samp * (1.0 - samp) * z * z);
        }
    }
    Ok(out)
}

fn scale_liability(s: &Matrix, liab: &[f64]) -> Matrix {
    let k = s.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = s[i][j] * liab[i].sqrt() * liab[j].sqrt();
        }
    }
    out
}

fn lower_triangle_ratio(a: &Matrix, b: &Matrix) -> Vec<f64> {
    let k = a.len();
    let mut out = Vec::new();
    for i in 0..k {
        for j in 0..=i {
            if b[i][j] != 0.0 {
                out.push(a[i][j] / b[i][j]);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

fn scale_v(v: &Matrix, scale: &[f64]) -> Matrix {
    let n = v.len();
    let mut diag = vec![0.0; n];
    for i in 0..n {
        diag[i] = v[i][i].abs().sqrt();
    }
    let mut scaled_diag = vec![0.0; n];
    for i in 0..n {
        scaled_diag[i] = diag[i] * scale.get(i).copied().unwrap_or(1.0);
    }
    let vcor = cov2cor(v, &diag);
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = vcor[i][j] * scaled_diag[i] * scaled_diag[j];
        }
    }
    out
}

fn cov2cor(v: &Matrix, diag: &[f64]) -> Matrix {
    let n = v.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if diag[i] != 0.0 && diag[j] != 0.0 {
                out[i][j] = v[i][j] / (diag[i] * diag[j]);
            }
        }
    }
    out
}

fn resolve_trait_names(config: &HdlConfig, n: usize) -> Result<Vec<String>> {
    if let Some(names) = &config.trait_names {
        if names.len() != n {
            return Err(anyhow::anyhow!(
                "trait.names length {} does not match traits length {}",
                names.len(),
                n
            ));
        }
        return Ok(names.clone());
    }
    Ok((1..=n).map(|i| format!("V{i}")).collect())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn ensure_column(df: &DataFrame, name: &str) -> Result<()> {
    if df.column(name).is_err() {
        return Err(anyhow::anyhow!("Missing required column {name}"));
    }
    Ok(())
}
