use anyhow::{Context, Result};
use ndarray::{Array2, ArrayBase, Ix2};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

use crate::matrix::to_array2;
use crate::types::{Estimation, LdscOutput, Matrix};
use lavaan as lavaan_crate;
use lavaan_crate::SemEngine as LavaanSemEngineTrait;
use lavaan_crate::stats::compute_q;
pub use lavaan_crate::{DefinedEstimate, ParamEstimate, SemFit, SemFitStats};

#[derive(Debug, Clone)]
pub struct CommonFactorResultRow {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub unstd_est: f64,
    pub unstd_se: f64,
    pub std_est: f64,
    pub std_se: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone)]
pub struct CommonFactorOutput {
    pub modelfit: SemFitStats,
    pub results: Vec<CommonFactorResultRow>,
    pub cfi_override: Option<f64>,
    pub trait_names: Vec<String>,
    pub s_original: Matrix,
    pub v_original: Matrix,
    pub s_smoothed: Matrix,
    pub v_smoothed: Matrix,
    pub smoothed_s: bool,
    pub smoothed_v: bool,
    pub ld_sdiff: f64,
    pub ld_sdiff2: f64,
    pub z_diff: f64,
}

#[derive(Debug, Clone)]
pub struct UserModelResultRow {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub unstd_est: f64,
    pub unstd_se: f64,
    pub std_est: f64,
    pub std_se: f64,
    pub std_all: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone)]
pub struct UserModelOutput {
    pub modelfit: SemFitStats,
    pub results: Vec<UserModelResultRow>,
    pub cfi_override: Option<f64>,
    pub trait_names: Vec<String>,
    pub s_original: Matrix,
    pub v_original: Matrix,
    pub s_smoothed: Matrix,
    pub v_smoothed: Matrix,
    pub smoothed_s: bool,
    pub smoothed_v: bool,
    pub ld_sdiff: f64,
    pub ld_sdiff2: f64,
    pub z_diff: f64,
}

#[derive(Debug, Clone)]
pub struct SemInput {
    pub s_full: Matrix,
    pub v_full: Matrix,
    pub model: String,
    pub model_table: Option<Vec<lavaan_crate::ParTableRow>>,
    pub wls_v: Option<Matrix>,
    pub estimation: Estimation,
    pub toler: Option<f64>,
    pub std_lv: bool,
    pub fix_measurement: bool,
    pub q_snp: bool,
    pub names: Vec<String>,
    pub n_obs: Option<f64>,
    pub optim_dx_tol: Option<f64>,
    pub optim_force_converged: bool,
    pub iter_max: Option<usize>,
    pub sample_cov_rescale: bool,
}

#[derive(Debug, Clone)]
pub struct SemOutput {
    pub est: f64,
    pub se: f64,
    pub se_c: f64,
    pub q: f64,
    pub fail: String,
    pub warning: String,
    pub implied: Matrix,
    pub residual: Matrix,
    pub params: Vec<SemParam>,
}

#[derive(Debug, Clone)]
pub struct SemParam {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub label: Option<String>,
    pub free: usize,
    pub est: f64,
    pub se: f64,
    pub se_c: f64,
}

pub trait SemEngine {
    fn fit(&self, input: &SemInput) -> Result<SemOutput>;
}

#[derive(Debug, Clone)]
pub struct LavaanSemEngine;

impl SemEngine for LavaanSemEngine {
    fn fit(&self, input: &SemInput) -> Result<SemOutput> {
        let (default_n_obs, default_sample_cov_rescale) = match input.estimation {
            Estimation::Dwls => (Some(2.0), false),
            Estimation::Ml => (Some(200.0), false),
        };
        let n_obs = input.n_obs.or(default_n_obs);
        let sample_cov_rescale = if input.sample_cov_rescale {
            true
        } else {
            default_sample_cov_rescale
        };
        let lavaan_input = lavaan_crate::SemInput {
            s: input.s_full.clone(),
            v: input.v_full.clone(),
            wls_v: input.wls_v.clone(),
            model: input.model.clone(),
            model_table: input.model_table.clone(),
            estimation: map_estimation(input.estimation),
            toler: input.toler,
            std_lv: input.std_lv,
            fix_measurement: input.fix_measurement,
            q_snp: input.q_snp,
            names: input.names.clone(),
            n_obs,
            optim_dx_tol: input.optim_dx_tol.or(Some(0.01)),
            optim_force_converged: input.optim_force_converged,
            iter_max: input.iter_max,
            sample_cov_rescale,
        };
        let engine = lavaan_crate::SemEngineImpl;
        let fit = engine.fit(&lavaan_input)?;
        let predictor = input.names.first().map(|s| s.as_str()).unwrap_or("SNP");
        let (se_all, se_c_all) = compute_sandwich_se(&fit, &input.v_full)?;
        let mut params = fit
            .params
            .iter()
            .map(|p| {
                let (se, se_c) = if p.free > 0 {
                    let idx = p.free - 1;
                    (
                        se_all.get(idx).copied().unwrap_or(f64::NAN),
                        se_c_all.get(idx).copied().unwrap_or(f64::NAN),
                    )
                } else {
                    (f64::NAN, f64::NAN)
                };
                SemParam {
                    lhs: p.lhs.clone(),
                    op: p.op.clone(),
                    rhs: p.rhs.clone(),
                    label: p.label.clone(),
                    free: p.free,
                    est: p.est,
                    se,
                    se_c,
                }
            })
            .collect::<Vec<_>>();

        for def in &fit.defined {
            params.push(SemParam {
                lhs: def.name.clone(),
                op: ":=".to_string(),
                rhs: def.expr.clone(),
                label: None,
                free: 0,
                est: def.est,
                se: def.se,
                se_c: def.se,
            });
        }

        let param = params.iter().find(|p| p.op == "~" && p.rhs == predictor);

        let (est, se, se_c) = if let Some(p) = param {
            (p.est, p.se, p.se_c)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        let q = compute_q(&input.s_full, &fit.implied, &input.v_full)?;

        Ok(SemOutput {
            est,
            se,
            se_c,
            q,
            fail: if fit.errors.is_empty() {
                "0".to_string()
            } else {
                fit.errors.join("; ")
            },
            warning: if fit.warnings.is_empty() {
                "0".to_string()
            } else {
                fit.warnings.join("; ")
            },
            implied: fit.implied.clone(),
            residual: fit.residual.clone(),
            params,
        })
    }
}

pub fn default_engine() -> Box<dyn SemEngine> {
    Box::new(LavaanSemEngine)
}

pub fn commonfactor(covstruc: &LdscOutput, estimation: Estimation) -> Result<CommonFactorOutput> {
    let s = &covstruc.s;
    let v = &covstruc.v;
    let k = s.len();
    if k < 3 {
        return Err(anyhow::anyhow!(
            "Common factor requires at least 3 traits; found {k}"
        ));
    }
    let z = k * (k + 1) / 2;
    if v.len() != z {
        return Err(anyhow::anyhow!(
            "V matrix must be {z}x{z} for {k} traits; found {}x{}",
            v.len(),
            v.first().map(|r| r.len()).unwrap_or(0)
        ));
    }

    let (s_smoothed, smoothed_s, ld_sdiff) = smooth_if_needed(s)?;
    let (v_smoothed, smoothed_v, ld_sdiff2) = smooth_if_needed(v)?;

    let z_diff = z_diff_metric(s, &s_smoothed, v, &v_smoothed);
    // R's .rearrange reorders V to lavaan's internal ordering.
    // Our lavaan implementation respects the input order, so no reordering is needed.
    let trait_names = if covstruc.trait_names.is_empty() {
        (1..=k).map(|i| format!("V{i}")).collect::<Vec<_>>()
    } else {
        covstruc.trait_names.clone()
    };
    let mut model = commonfactor_model_string(&trait_names);
    let mut fit = fit_sem(FitSemConfig {
        s: &s_smoothed,
        v: &v_smoothed,
        wls_v: None,
        model: &model,
        names: &trait_names,
        estimation,
        std_lv: false,
        optim_force_converged: false,
    })?;
    if !fit.converged || !fit.errors.is_empty() {
        model = add_resid_bounds(&model, &trait_names);
        fit = fit_sem(FitSemConfig {
            s: &s_smoothed,
            v: &v_smoothed,
            wls_v: None,
            model: &model,
            names: &trait_names,
            estimation,
            std_lv: false,
            optim_force_converged: false,
        })?;
    }
    let (s_stand, v_stand, wls_v_stand) = standardize_covariance(&s_smoothed, &v_smoothed)?;
    let fit_stand = fit_sem(FitSemConfig {
        s: &s_stand,
        v: &v_stand,
        wls_v: Some(wls_v_stand),
        model: &model,
        names: &trait_names,
        estimation,
        std_lv: false,
        optim_force_converged: false,
    })?;
    let results = build_commonfactor_results(&fit, &fit_stand)?;

    let cfi_override = compute_cfi_r(
        &s_smoothed,
        &v_smoothed,
        estimation,
        fit.stats.chisq,
        fit.stats.df,
        &trait_names,
    )
    .ok();

    Ok(CommonFactorOutput {
        modelfit: fit.stats.clone(),
        results,
        cfi_override,
        trait_names,
        s_original: s.clone(),
        v_original: v.clone(),
        s_smoothed,
        v_smoothed,
        smoothed_s,
        smoothed_v,
        ld_sdiff,
        ld_sdiff2,
        z_diff,
    })
}

pub fn usermodel_fit(
    covstruc: &LdscOutput,
    model: &str,
    estimation: Estimation,
) -> Result<lavaan_crate::SemFit> {
    fit_sem(FitSemConfig {
        s: &covstruc.s,
        v: &covstruc.v,
        wls_v: None,
        model,
        names: &covstruc.trait_names,
        estimation,
        std_lv: false,
        optim_force_converged: false,
    })
}

pub fn usermodel(
    covstruc: &LdscOutput,
    model: &str,
    estimation: Estimation,
    std_lv: bool,
    cfi_calc: bool,
) -> Result<UserModelOutput> {
    let (covstruc, trait_names) = subset_covstruc(covstruc, model)?;
    let (s_smoothed, smoothed_s, ld_sdiff) = smooth_if_needed(&covstruc.s)?;
    let (v_smoothed, smoothed_v, ld_sdiff2) = smooth_if_needed(&covstruc.v)?;
    let z_diff = z_diff_metric(&covstruc.s, &s_smoothed, &covstruc.v, &v_smoothed);
    // R's .rearrange reorders V to lavaan's internal ordering.
    // Our lavaan implementation respects the input order, so no reordering is needed.

    let mut model_used = model.to_string();
    let mut fit = fit_sem(FitSemConfig {
        s: &s_smoothed,
        v: &v_smoothed,
        wls_v: None,
        model: &model_used,
        names: &trait_names,
        estimation,
        std_lv,
        optim_force_converged: false,
    })?;
    if !fit.converged || !fit.errors.is_empty() {
        model_used = add_resid_bounds(&model_used, &trait_names);
        fit = fit_sem(FitSemConfig {
            s: &s_smoothed,
            v: &v_smoothed,
            wls_v: None,
            model: &model_used,
            names: &trait_names,
            estimation,
            std_lv,
            optim_force_converged: false,
        })?;
    }
    let (s_stand, v_stand, wls_v_stand) = standardize_covariance(&s_smoothed, &v_smoothed)?;
    let fit_stand = fit_sem(FitSemConfig {
        s: &s_stand,
        v: &v_stand,
        wls_v: Some(wls_v_stand),
        model: &model_used,
        names: &trait_names,
        estimation,
        std_lv,
        optim_force_converged: false,
    })?;
    let results = build_usermodel_results(&fit, &fit_stand)?;

    let cfi_override = if cfi_calc {
        compute_cfi_r(
            &s_smoothed,
            &v_smoothed,
            estimation,
            fit.stats.chisq,
            fit.stats.df,
            &trait_names,
        )
        .ok()
    } else {
        None
    };

    Ok(UserModelOutput {
        modelfit: fit.stats.clone(),
        results,
        cfi_override,
        trait_names,
        s_original: covstruc.s.clone(),
        v_original: covstruc.v.clone(),
        s_smoothed,
        v_smoothed,
        smoothed_s,
        smoothed_v,
        ld_sdiff,
        ld_sdiff2,
        z_diff,
    })
}

pub fn commonfactor_output_tables(out: &CommonFactorOutput) -> Result<(DataFrame, DataFrame)> {
    let (modelfit, results) = build_commonfactor_tables(out)?;
    Ok((modelfit, results))
}

pub fn usermodel_output_tables(out: &UserModelOutput) -> Result<(DataFrame, DataFrame)> {
    let (modelfit, results) = build_usermodel_tables(out)?;
    Ok((modelfit, results))
}

pub(crate) fn smooth_if_needed(matrix: &Matrix) -> Result<(Matrix, bool, f64)> {
    let a = to_array2(matrix)?;
    let (eigvals, eigvecs) = a.eigh(UPLO::Lower)?;
    let min_eig = eigvals.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig > 0.0 && min_eig.is_finite() {
        return Ok((matrix.clone(), false, 0.0));
    }

    // TODO(Matrix::nearPD): Replace eigenvalue clipping with a true nearPD algorithm if available.
    let eps = 1e-8;
    let adj_vals = eigvals.mapv(|v| if v.is_finite() && v > eps { v } else { eps });
    let diag = Array2::from_diag(&adj_vals);
    let smoothed = eigvecs.dot(&diag).dot(&eigvecs.t());
    let smoothed_vec = from_array2(&smoothed);
    let diff = max_abs_diff(matrix, &smoothed_vec);
    Ok((smoothed_vec, true, diff))
}

fn z_diff_metric(s_orig: &Matrix, s_smooth: &Matrix, v_orig: &Matrix, v_smooth: &Matrix) -> f64 {
    let k = s_orig.len();
    let se_pre = lower_triangle_se(v_orig, k);
    let se_post = lower_triangle_se(v_smooth, k);
    if se_pre.is_empty() || se_post.is_empty() {
        return 0.0;
    }
    let z_pre = elementwise_div(s_orig, &se_pre);
    let z_post = elementwise_div(s_smooth, &se_post);
    max_abs_diff(&z_pre, &z_post)
}

#[allow(clippy::needless_range_loop)]
fn lower_triangle_se(v: &Matrix, k: usize) -> Matrix {
    if k == 0 {
        return vec![];
    }
    let mut se = vec![vec![0.0; k]; k];
    let mut idx = 0usize;
    for j in 0..k {
        for i in j..k {
            let val = v
                .get(idx)
                .and_then(|row| row.get(idx))
                .copied()
                .unwrap_or(0.0);
            if val.is_finite() && val >= 0.0 {
                se[i][j] = val.sqrt();
            }
            idx += 1;
        }
    }
    se
}

fn elementwise_div(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let denom = b[i][j];
            out[i][j] = if denom != 0.0 { a[i][j] / denom } else { 0.0 };
        }
    }
    out
}

fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
    let mut max = 0.0;
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let diff = (a[i][j] - b[i][j]).abs();
            if diff > max {
                max = diff;
            }
        }
    }
    max
}

fn from_array2(matrix: &ArrayBase<impl ndarray::Data<Elem = f64>, Ix2>) -> Matrix {
    let (n, m) = matrix.dim();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            out[i][j] = matrix[(i, j)];
        }
    }
    out
}

fn diag_sqrt(cov: &Array2<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(cov.dim().0);
    for i in 0..cov.dim().0 {
        let v = cov[(i, i)];
        out.push(if v.is_finite() && v >= 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        });
    }
    out
}

fn compute_sandwich_se(fit: &SemFit, gamma: &Matrix) -> Result<(Vec<f64>, Vec<f64>)> {
    let delta = to_array2(&fit.delta)?;
    let w = to_array2(&fit.wls_v)?;
    let v = to_array2(gamma)?;
    let bread = delta.t().dot(&w).dot(&delta).inv()?;
    let se = diag_sqrt(&bread);

    let lettuce = w.dot(&delta);
    let cov = bread.dot(&lettuce.t().dot(&v).dot(&lettuce)).dot(&bread);
    let se_c = diag_sqrt(&cov);
    Ok((se, se_c))
}

fn map_estimation(est: Estimation) -> lavaan_crate::Estimation {
    match est {
        Estimation::Dwls => lavaan_crate::Estimation::Dwls,
        Estimation::Ml => lavaan_crate::Estimation::Ml,
    }
}

struct FitSemConfig<'a> {
    s: &'a Matrix,
    v: &'a Matrix,
    wls_v: Option<Matrix>,
    model: &'a str,
    names: &'a [String],
    estimation: Estimation,
    std_lv: bool,
    optim_force_converged: bool,
}

fn fit_sem(config: FitSemConfig<'_>) -> Result<lavaan_crate::SemFit> {
    let (n_obs, sample_cov_rescale) = match config.estimation {
        Estimation::Dwls => (Some(2.0), false),
        Estimation::Ml => (Some(200.0), false),
    };
    let input = lavaan_crate::SemInput {
        s: config.s.clone(),
        v: config.v.clone(),
        wls_v: config.wls_v,
        model: config.model.to_string(),
        model_table: None,
        estimation: map_estimation(config.estimation),
        toler: Some(f64::EPSILON),
        std_lv: config.std_lv,
        fix_measurement: false,
        q_snp: false,
        names: config.names.to_vec(),
        n_obs,
        optim_dx_tol: Some(0.01),
        optim_force_converged: config.optim_force_converged,
        iter_max: None,
        sample_cov_rescale,
    };
    let engine = lavaan_crate::SemEngineImpl;
    engine.fit(&input)
}

fn commonfactor_model_string(trait_names: &[String]) -> String {
    if trait_names.is_empty() {
        return "F1 =~ ".to_string();
    }
    let mut model = format!("F1 =~ start(0.1)*{}", trait_names[0]);
    for name in trait_names.iter().skip(1) {
        model.push_str(" + ");
        model.push_str(name);
    }
    model.push_str("\nF1 ~~ 1*F1");
    model
}

fn add_resid_bounds(model: &str, trait_names: &[String]) -> String {
    let mut out = model.to_string();
    let mut used = std::collections::HashSet::new();
    let mut counter = 0usize;
    for name in trait_names {
        loop {
            let candidate = label_from_index(counter);
            counter += 1;
            if !model.contains(&candidate) && !used.contains(&candidate) {
                used.insert(candidate.clone());
                out.push_str(&format!(
                    "\n{name} ~~ {candidate}*{name}\n{candidate} > .0001"
                ));
                break;
            }
        }
    }
    out
}

fn label_from_index(mut idx: usize) -> String {
    let letters = b"abcdefghijklmnopqrstuvwxyz";
    let mut out = String::with_capacity(4);
    for _ in 0..4 {
        let c = letters[idx % 26] as char;
        out.push(c);
        idx /= 26;
    }
    out
}

fn build_commonfactor_results(
    unstd: &SemFit,
    stand: &SemFit,
) -> Result<Vec<CommonFactorResultRow>> {
    let mut std_map: HashMap<String, (f64, f64)> = HashMap::new();
    for p in stand.params.iter().filter(|p| p.free > 0) {
        let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
        std_map.insert(key, (p.est, p.se));
    }
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let mut out = Vec::new();
    for p in unstd.params.iter().filter(|p| p.free > 0) {
        let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
        let (std_est, std_se) = std_map.get(&key).copied().unwrap_or((f64::NAN, f64::NAN));
        let p_value = z_to_pvalue(&normal, p.est, p.se);
        out.push(CommonFactorResultRow {
            lhs: p.lhs.clone(),
            op: p.op.clone(),
            rhs: p.rhs.clone(),
            unstd_est: p.est,
            unstd_se: p.se,
            std_est,
            std_se,
            p_value,
        });
    }
    Ok(out)
}

fn build_usermodel_results(unstd: &SemFit, stand: &SemFit) -> Result<Vec<UserModelResultRow>> {
    let mut std_map: HashMap<String, (f64, f64, f64)> = HashMap::new();
    for p in stand.params.iter().filter(|p| p.free > 0) {
        let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
        let std_all = p.est_std_all.unwrap_or(f64::NAN);
        std_map.insert(key, (p.est, p.se, std_all));
    }
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let mut out = Vec::new();
    for p in unstd.params.iter().filter(|p| p.free > 0) {
        let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
        let (std_est, std_se, std_all) =
            std_map
                .get(&key)
                .copied()
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        let p_value = z_to_pvalue(&normal, p.est, p.se);
        out.push(UserModelResultRow {
            lhs: p.lhs.clone(),
            op: p.op.clone(),
            rhs: p.rhs.clone(),
            unstd_est: p.est,
            unstd_se: p.se,
            std_est,
            std_se,
            std_all,
            p_value,
        });
    }
    Ok(out)
}

fn z_to_pvalue(normal: &Normal, est: f64, se: f64) -> f64 {
    if !est.is_finite() || !se.is_finite() || se == 0.0 {
        return f64::NAN;
    }
    let z = (est / se).abs();
    let mut p = 2.0 * (1.0 - normal.cdf(z));
    if p == 0.0 {
        p = 5e-300;
    }
    p
}

fn build_commonfactor_tables(out: &CommonFactorOutput) -> Result<(DataFrame, DataFrame)> {
    let modelfit = modelfit_table(&out.modelfit, out.cfi_override)?;
    let mut lhs = Vec::with_capacity(out.results.len());
    let mut op = Vec::with_capacity(out.results.len());
    let mut rhs = Vec::with_capacity(out.results.len());
    let mut unstd_est = Vec::with_capacity(out.results.len());
    let mut unstd_se = Vec::with_capacity(out.results.len());
    let mut std_est = Vec::with_capacity(out.results.len());
    let mut std_se = Vec::with_capacity(out.results.len());
    let mut p_value = Vec::with_capacity(out.results.len());
    for row in &out.results {
        lhs.push(row.lhs.clone());
        op.push(row.op.clone());
        rhs.push(row.rhs.clone());
        unstd_est.push(row.unstd_est);
        unstd_se.push(row.unstd_se);
        std_est.push(row.std_est);
        std_se.push(row.std_se);
        p_value.push(row.p_value);
    }
    let results = DataFrame::new(
        out.results.len(),
        vec![
            Series::new("lhs".into(), lhs).into(),
            Series::new("op".into(), op).into(),
            Series::new("rhs".into(), rhs).into(),
            Series::new("Unstandardized_Estimate".into(), unstd_est).into(),
            Series::new("Unstandardized_SE".into(), unstd_se).into(),
            Series::new("Standardized_Est".into(), std_est).into(),
            Series::new("Standardized_SE".into(), std_se).into(),
            Series::new("p_value".into(), p_value).into(),
        ],
    )?;
    Ok((modelfit, results))
}

fn build_usermodel_tables(out: &UserModelOutput) -> Result<(DataFrame, DataFrame)> {
    let modelfit = modelfit_table(&out.modelfit, out.cfi_override)?;
    let mut lhs = Vec::with_capacity(out.results.len());
    let mut op = Vec::with_capacity(out.results.len());
    let mut rhs = Vec::with_capacity(out.results.len());
    let mut unstd_est = Vec::with_capacity(out.results.len());
    let mut unstd_se = Vec::with_capacity(out.results.len());
    let mut std_est = Vec::with_capacity(out.results.len());
    let mut std_se = Vec::with_capacity(out.results.len());
    let mut std_all = Vec::with_capacity(out.results.len());
    let mut p_value = Vec::with_capacity(out.results.len());
    for row in &out.results {
        lhs.push(row.lhs.clone());
        op.push(row.op.clone());
        rhs.push(row.rhs.clone());
        unstd_est.push(row.unstd_est);
        unstd_se.push(row.unstd_se);
        std_est.push(row.std_est);
        std_se.push(row.std_se);
        std_all.push(row.std_all);
        p_value.push(row.p_value);
    }
    let results = DataFrame::new(
        out.results.len(),
        vec![
            Series::new("lhs".into(), lhs).into(),
            Series::new("op".into(), op).into(),
            Series::new("rhs".into(), rhs).into(),
            Series::new("Unstand_Est".into(), unstd_est).into(),
            Series::new("Unstand_SE".into(), unstd_se).into(),
            Series::new("STD_Genotype".into(), std_est).into(),
            Series::new("STD_Genotype_SE".into(), std_se).into(),
            Series::new("STD_All".into(), std_all).into(),
            Series::new("p_value".into(), p_value).into(),
        ],
    )?;
    Ok((modelfit, results))
}

fn modelfit_table(stats: &SemFitStats, cfi_override: Option<f64>) -> Result<DataFrame> {
    let df_val = stats.df as f64;
    let mut chisq = stats.chisq;
    let mut aic = stats.aic;
    let mut p_chisq = stats.p_chisq;
    if stats.df == 0 {
        chisq = f64::NAN;
        aic = f64::NAN;
        p_chisq = f64::NAN;
    }
    let cfi_val = cfi_override.unwrap_or(stats.cfi);
    let modelfit = DataFrame::new(
        1,
        vec![
            Series::new("chisq".into(), vec![chisq]).into(),
            Series::new("df".into(), vec![df_val]).into(),
            Series::new("p_chisq".into(), vec![p_chisq]).into(),
            Series::new("AIC".into(), vec![aic]).into(),
            Series::new("CFI".into(), vec![cfi_val]).into(),
            Series::new("SRMR".into(), vec![stats.srmr]).into(),
        ],
    )?;
    Ok(modelfit)
}

fn compute_cfi_r(
    s: &Matrix,
    v: &Matrix,
    estimation: Estimation,
    q: f64,
    df: i64,
    trait_names: &[String],
) -> Result<f64> {
    let k = s.len();
    if k == 0 {
        return Ok(f64::NAN);
    }
    let z = k * (k + 1) / 2;
    if v.len() != z {
        return Ok(f64::NAN);
    }
    let model_cfi = null_model_string(trait_names);
    let wls_v = if matches!(estimation, Estimation::Dwls) {
        Some(wls_v_diag_inverse(v))
    } else {
        None
    };
    let fit_cfi = fit_sem(FitSemConfig {
        s,
        v,
        wls_v: wls_v.clone(),
        model: &model_cfi,
        names: trait_names,
        estimation,
        std_lv: false,
        optim_force_converged: false,
    })?;
    if !fit_cfi.converged || !fit_cfi.errors.is_empty() {
        return Ok(f64::NAN);
    }
    let model_table = cfi_model_table(&fit_cfi.par_table, trait_names);
    let fit_q = fit_sem_table(FitSemTableConfig {
        s,
        v,
        wls_v,
        model_table,
        names: trait_names,
        estimation,
        std_lv: false,
    })?;
    if !fit_q.converged || !fit_q.errors.is_empty() {
        return Ok(f64::NAN);
    }
    let eta = free_param_vector(&fit_q.params, fit_q.npar);
    let q_cfi = quad_form_vinv(&eta, &fit_q.vcov)?;
    let df_cfi = (z - k) as f64;
    let denom = q_cfi - df_cfi;
    if !denom.is_finite() || denom == 0.0 {
        return Ok(f64::NAN);
    }
    let mut cfi = ((q_cfi - df_cfi) - (q - df as f64)) / denom;
    if cfi > 1.0 {
        cfi = 1.0;
    }
    Ok(cfi)
}

#[allow(clippy::needless_range_loop)]
fn null_model_string(trait_names: &[String]) -> String {
    let k = trait_names.len();
    let mut model = String::new();
    for name in trait_names {
        model.push_str(&format!("{name} ~~ {name}\n"));
    }
    for (idx, name) in trait_names.iter().enumerate() {
        model.push_str(&format!("VF{} =~ 1*{}\n", idx + 1, name));
    }
    if k > 1 {
        for i in 0..(k - 1) {
            let mut line = format!("{} ~~ 0*{}", trait_names[i], trait_names[i + 1]);
            for j in (i + 2)..k {
                line.push_str(&format!(" + 0*{}", trait_names[j]));
            }
            model.push_str(&format!("{line}\n"));
        }
        for i in 0..(k - 1) {
            let mut line = format!("VF{} ~~ 0*VF{}", i + 1, i + 2);
            for j in (i + 2)..k {
                line.push_str(&format!(" + 0*VF{}", j + 1));
            }
            model.push_str(&format!("{line}\n"));
        }
    }
    for idx in 0..k {
        model.push_str(&format!("VF{} ~~ 0*VF{}\n", idx + 1, idx + 1));
    }
    model
}

fn wls_v_diag_inverse(v: &Matrix) -> Matrix {
    let n = v.len();
    let mut out = vec![vec![0.0; n]; n];
    for (i, row) in v.iter().enumerate() {
        let val = row[i];
        if val.is_finite() && val != 0.0 {
            out[i][i] = 1.0 / val;
        }
    }
    out
}

struct FitSemTableConfig<'a> {
    s: &'a Matrix,
    v: &'a Matrix,
    wls_v: Option<Matrix>,
    model_table: Vec<lavaan_crate::ParTableRow>,
    names: &'a [String],
    estimation: Estimation,
    std_lv: bool,
}

fn fit_sem_table(config: FitSemTableConfig<'_>) -> Result<lavaan_crate::SemFit> {
    let (n_obs, sample_cov_rescale) = match config.estimation {
        Estimation::Dwls => (Some(2.0), false),
        Estimation::Ml => (Some(200.0), false),
    };
    let input = lavaan_crate::SemInput {
        s: config.s.clone(),
        v: config.v.clone(),
        wls_v: config.wls_v,
        model: String::new(),
        model_table: Some(config.model_table),
        estimation: map_estimation(config.estimation),
        toler: Some(f64::EPSILON),
        std_lv: config.std_lv,
        fix_measurement: false,
        q_snp: false,
        names: config.names.to_vec(),
        n_obs,
        optim_dx_tol: Some(0.01),
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale,
    };
    let engine = lavaan_crate::SemEngineImpl;
    engine.fit(&input)
}

fn cfi_model_table(
    par_table: &[lavaan_crate::ParTableRow],
    names: &[String],
) -> Vec<lavaan_crate::ParTableRow> {
    let mut out = par_table.to_vec();
    let mut free_map = HashMap::new();
    let mut idx = 1usize;
    for j in 0..names.len() {
        for i in j..names.len() {
            let lhs = &names[i];
            let rhs = &names[j];
            free_map.insert(format!("{lhs}~~{rhs}"), idx);
            free_map.insert(format!("{rhs}~~{lhs}"), idx);
            idx += 1;
        }
    }
    for row in out.iter_mut() {
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        if row.op == "~~" && free_map.contains_key(&key) {
            row.free = *free_map.get(&key).unwrap_or(&0);
            row.ustart = row.est;
        } else {
            row.free = 0;
            row.ustart = row.est;
        }
    }
    out
}

fn free_param_vector(params: &[ParamEstimate], npar: usize) -> Vec<f64> {
    let mut eta = vec![0.0; npar];
    for p in params {
        if p.free > 0 && p.free <= npar {
            eta[p.free - 1] = p.est;
        }
    }
    eta
}

fn quad_form_vinv(vec: &[f64], v: &Matrix) -> Result<f64> {
    let arr = to_array2(v)?;
    let (eigvals, eigvecs) = arr.eigh(UPLO::Lower)?;
    let mut inv_vals = eigvals.to_vec();
    for v in &mut inv_vals {
        *v = if *v > 0.0 { 1.0 / *v } else { f64::INFINITY };
    }
    let inv_diag = Array2::from_diag(&ndarray::Array1::from_vec(inv_vals));
    let v_inv = eigvecs.dot(&inv_diag).dot(&eigvecs.t());

    let mut tmp = vec![0.0; vec.len()];
    for i in 0..vec.len() {
        let mut sum = 0.0;
        for j in 0..vec.len() {
            sum += v_inv[(i, j)] * vec[j];
        }
        tmp[i] = sum;
    }
    let mut out = 0.0;
    for i in 0..vec.len() {
        out += vec[i] * tmp[i];
    }
    Ok(out)
}

fn standardize_covariance(s: &Matrix, v: &Matrix) -> Result<(Matrix, Matrix, Matrix)> {
    let k = s.len();
    if k == 0 {
        return Err(anyhow::anyhow!("S matrix must not be empty"));
    }
    if v.len() != k * (k + 1) / 2 {
        return Err(anyhow::anyhow!(
            "V matrix must be {}x{} for {k} traits; found {}x{}",
            k * (k + 1) / 2,
            k * (k + 1) / 2,
            v.len(),
            v.first().map(|row| row.len()).unwrap_or(0)
        ));
    }
    let mut inv_sqrt = Vec::with_capacity(k);
    for (i, row) in s.iter().enumerate() {
        let v = row[i];
        if !v.is_finite() || v <= 0.0 {
            return Err(anyhow::anyhow!(
                "S matrix diagonal must be positive for standardization"
            ));
        }
        inv_sqrt.push(1.0 / v.sqrt());
    }
    let ratio = outer_from_vec(&inv_sqrt);
    let s_stand = elementwise_mul(s, &ratio);
    let scale_o = lower_triangle_values(&ratio);
    let mut dvcov = Vec::with_capacity(v.len());
    for (i, row) in v.iter().enumerate() {
        let val = row[i];
        dvcov.push(if val.is_finite() && val >= 0.0 {
            val.sqrt()
        } else {
            0.0
        });
    }
    let mut dvcovl = Vec::with_capacity(dvcov.len());
    for (i, base) in dvcov.iter().enumerate() {
        let mut val = base * scale_o[i];
        if !val.is_finite() {
            val = 0.0;
        }
        dvcovl.push(val);
    }
    let vcor = cov2cor(v);
    let v_stand = scale_sampling_covariance(&vcor, &dvcovl);
    let mut wls_v = vec![vec![0.0; v_stand.len()]; v_stand.len()];
    for (i, row) in v_stand.iter().enumerate() {
        let mut diag = row[i];
        if !diag.is_finite() || diag == 0.0 {
            diag = 2e-9;
        }
        wls_v[i][i] = 1.0 / diag;
    }
    Ok((s_stand, v_stand, wls_v))
}

fn outer_from_vec(values: &[f64]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; values.len()]; values.len()];
    for i in 0..values.len() {
        for j in 0..values.len() {
            out[i][j] = values[i] * values[j];
        }
    }
    out
}

fn elementwise_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            out[i][j] = a[i][j] * b[i][j];
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn lower_triangle_values(matrix: &Matrix) -> Vec<f64> {
    let n = matrix.len();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in j..n {
            out.push(matrix[i][j]);
        }
    }
    out
}

fn cov2cor(cov: &Matrix) -> Matrix {
    let n = cov.len();
    let mut out = vec![vec![0.0; n]; n];
    let mut sd = vec![0.0; n];
    for i in 0..n {
        let v = cov[i][i];
        sd[i] = if v.is_finite() && v > 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        };
    }
    for i in 0..n {
        for j in 0..n {
            let denom = sd[i] * sd[j];
            out[i][j] = if denom != 0.0 && denom.is_finite() {
                cov[i][j] / denom
            } else {
                0.0
            };
        }
    }
    out
}

fn scale_sampling_covariance(v: &Matrix, scale: &[f64]) -> Matrix {
    let n = scale.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = v[i][j] * scale[i] * scale[j];
        }
    }
    out
}

fn subset_covstruc(covstruc: &LdscOutput, model: &str) -> Result<(LdscOutput, Vec<String>)> {
    if covstruc.trait_names.is_empty() {
        return Err(anyhow::anyhow!(
            "trait names are required to subset model variables"
        ));
    }
    let mut keep = Vec::new();
    for (idx, name) in covstruc.trait_names.iter().enumerate() {
        if model_contains_name(model, name) {
            keep.push(idx);
        }
    }
    if keep.is_empty() {
        return Err(anyhow::anyhow!(
            "None of the trait names in the LDSC output match names in the model"
        ));
    }
    let mut s_sub = vec![vec![0.0; keep.len()]; keep.len()];
    for (i_out, &i) in keep.iter().enumerate() {
        for (j_out, &j) in keep.iter().enumerate() {
            s_sub[i_out][j_out] = covstruc.s[i][j];
        }
    }
    let v_sub = subset_sampling_covariance(&covstruc.v, &keep, covstruc.s.len())?;
    let trait_names = keep
        .iter()
        .map(|&i| covstruc.trait_names[i].clone())
        .collect::<Vec<_>>();
    let mut out = covstruc.clone();
    out.s = s_sub;
    out.v = v_sub;
    out.trait_names = trait_names.clone();
    out.s_stand = None;
    out.v_stand = None;
    Ok((out, trait_names))
}

fn model_contains_name(model: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    let mut start = 0usize;
    while let Some(pos) = model[start..].find(name) {
        let idx = start + pos;
        let end = idx + name.len();
        let before = model[..idx].chars().last();
        let after = model[end..].chars().next();
        let before_ok = before.is_none_or(|c| !is_word(c));
        let after_ok = after.is_none_or(|c| !is_word(c));
        if before_ok && after_ok {
            return true;
        }
        start = end;
    }
    false
}

fn subset_sampling_covariance(v: &Matrix, keep: &[usize], k: usize) -> Result<Matrix> {
    let z = k * (k + 1) / 2;
    if v.len() != z {
        return Err(anyhow::anyhow!(
            "V matrix must be {z}x{z} for {k} traits; found {}x{}",
            v.len(),
            v.first().map(|row| row.len()).unwrap_or(0)
        ));
    }
    let mut indices = Vec::new();
    for (pos_j, &j) in keep.iter().enumerate() {
        for &i in keep.iter().skip(pos_j) {
            indices.push(vech_index(i, j, k));
        }
    }
    let m = indices.len();
    let mut out = vec![vec![0.0; m]; m];
    for (i_out, &i_idx) in indices.iter().enumerate() {
        for (j_out, &j_idx) in indices.iter().enumerate() {
            out[i_out][j_out] = v[i_idx][j_idx];
        }
    }
    Ok(out)
}

fn vech_index(i: usize, j: usize, k: usize) -> usize {
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    col * k - (col * (col.saturating_sub(1))) / 2 + (row - col)
}
