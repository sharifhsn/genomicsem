use anyhow::{Context, Result};
use polars::prelude::*;
use rayon::prelude::*;

use crate::parallel::{collect_results, resolve_threads, run_in_pool};
use crate::sem::{LavaanSemEngine, SemEngine, SemInput};
use crate::types::{Estimation, GenomicControl, LdscOutput, Matrix, SumstatsTable};
use lavaan as lavaan_crate;
use lavaan_crate::SemEngine as LavaanSemEngineTrait;
use lavaan_crate::parser::ModelOp;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

#[derive(Debug, Clone)]
pub struct CommonFactorGwasConfig {
    pub estimation: Estimation,
    pub cores: Option<usize>,
    pub toler: Option<f64>,
    pub snp_se: Option<f64>,
    pub parallel: bool,
    pub gc: GenomicControl,
    pub mpi: bool,
    pub twas: bool,
    pub smooth_check: bool,
}

#[derive(Debug, Clone)]
pub struct CommonFactorPrepared {
    pub trait_names: Vec<String>,
    pub s_ld: Matrix,
    pub v_ld: Matrix,
    pub i_ld: Matrix,
    pub beta: Vec<Vec<f64>>,
    pub se: Vec<Vec<f64>>,
    pub snp_ids: Vec<String>,
    pub a1: Vec<String>,
    pub a2: Vec<String>,
    pub maf: Vec<f64>,
    pub var_snp: Vec<f64>,
    pub var_snp_se2: f64,
    pub panel: Option<Vec<String>>,
    pub gene: Option<Vec<String>>,
    pub hsq: Option<Vec<f64>>,
    pub twas: bool,
    pub smooth_check: bool,
    pub meta: Option<DataFrame>,
}

#[derive(Debug, Clone)]
pub struct CommonFactorGwasRow {
    pub i: usize,
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub est: f64,
    pub se: f64,
    pub se_c: f64,
    pub q: f64,
    pub fail: String,
    pub warning: String,
    pub z_smooth: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CommonFactorGwasOutput {
    pub prepared: CommonFactorPrepared,
    pub results: Vec<CommonFactorGwasRow>,
}

#[derive(Debug, Clone)]
pub struct UserGwasRow {
    pub i: usize,
    pub param_index: usize,
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub est: f64,
    pub se: f64,
    pub se_c: f64,
    pub q: f64,
    pub q_snp: Option<f64>,
    pub q_snp_df: Option<f64>,
    pub q_snp_pval: Option<f64>,
    pub fail: String,
    pub warning: String,
    pub z_smooth: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct UserGwasPrepared {
    pub trait_names: Vec<String>,
    pub s_ld: Matrix,
    pub v_ld: Matrix,
    pub i_ld: Matrix,
    pub beta: Vec<Vec<f64>>,
    pub se: Vec<Vec<f64>>,
    pub snp_ids: Vec<String>,
    pub a1: Vec<String>,
    pub a2: Vec<String>,
    pub maf: Vec<f64>,
    pub var_snp: Vec<f64>,
    pub var_snp_se2: f64,
    pub model: String,
    pub model_table: Option<Vec<lavaan_crate::ParTableRow>>,
    pub predictor: String,
    pub q_snp: bool,
    pub panel: Option<Vec<String>>,
    pub gene: Option<Vec<String>>,
    pub hsq: Option<Vec<f64>>,
    pub twas: bool,
    pub meta: Option<DataFrame>,
}

#[derive(Debug, Clone)]
pub struct UserGwasOutput {
    pub prepared: UserGwasPrepared,
    pub results: Vec<UserGwasRow>,
}

pub fn commonfactor_gwas_output_table(out: &CommonFactorGwasOutput) -> Result<DataFrame> {
    let rows: Vec<GwasRowView<'_>> = out
        .results
        .iter()
        .map(|r| GwasRowView {
            i: r.i,
            lhs: &r.lhs,
            op: &r.op,
            rhs: &r.rhs,
            est: r.est,
            se: r.se,
            se_c: r.se_c,
            q: r.q,
            q_snp: None,
            q_snp_df: None,
            q_snp_pval: None,
            fail: &r.fail,
            warning: &r.warning,
            z_smooth: r.z_smooth,
        })
        .collect();
    let mut df = build_gwas_output_table(out.prepared.meta.as_ref(), &rows)?;
    if df.column("se").is_ok() {
        df.drop_in_place("se")?;
    }
    let n = df.height();
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let z_est = out
        .results
        .iter()
        .map(|r| {
            if r.se_c.is_finite() && r.se_c != 0.0 {
                r.est / r.se_c
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let p_est = z_est
        .iter()
        .map(|z| {
            if z.is_finite() {
                2.0 * (1.0 - normal.cdf(z.abs()))
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let q_df_val = if out.prepared.trait_names.is_empty() {
        f64::NAN
    } else {
        (out.prepared.trait_names.len() as f64) - 1.0
    };
    let chi = if q_df_val.is_finite() && q_df_val > 0.0 {
        Some(ChiSquared::new(q_df_val)?)
    } else {
        None
    };
    let q_pval = out
        .results
        .iter()
        .map(|r| {
            if let Some(chi) = &chi {
                if r.q.is_finite() {
                    1.0 - chi.cdf(r.q)
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    df.with_column(Series::new("Z_Estimate".into(), z_est).into())?;
    df.with_column(Series::new("Pval_Estimate".into(), p_est).into())?;
    df.with_column(Series::new("Q_df".into(), vec![q_df_val; n]).into())?;
    df.with_column(Series::new("Q_pval".into(), q_pval).into())?;
    if out.prepared.twas {
        let rhs = vec!["Gene".to_string(); n];
        df.with_column(Series::new("rhs".into(), rhs).into())?;
    }
    reorder_commonfactor_columns(
        &mut df,
        out.prepared.twas,
        out.prepared.smooth_check,
        out.prepared
            .meta
            .as_ref()
            .map(|m| m.get_column_names().iter().map(|s| s.as_str()).collect())
            .unwrap_or_default(),
    )?;
    Ok(df)
}

pub fn user_gwas_output_table(out: &UserGwasOutput) -> Result<DataFrame> {
    let rows: Vec<GwasRowView<'_>> = out
        .results
        .iter()
        .map(|r| GwasRowView {
            i: r.i,
            lhs: &r.lhs,
            op: &r.op,
            rhs: &r.rhs,
            est: r.est,
            se: r.se,
            se_c: r.se_c,
            q: r.q,
            q_snp: r.q_snp,
            q_snp_df: r.q_snp_df,
            q_snp_pval: r.q_snp_pval,
            fail: &r.fail,
            warning: &r.warning,
            z_smooth: r.z_smooth,
        })
        .collect();
    build_gwas_output_table(out.prepared.meta.as_ref(), &rows)
}

#[derive(Clone)]
struct GwasRowView<'a> {
    i: usize,
    lhs: &'a str,
    op: &'a str,
    rhs: &'a str,
    est: f64,
    se: f64,
    se_c: f64,
    q: f64,
    q_snp: Option<f64>,
    q_snp_df: Option<f64>,
    q_snp_pval: Option<f64>,
    fail: &'a str,
    warning: &'a str,
    z_smooth: Option<f64>,
}

fn build_gwas_output_table(
    meta: Option<&DataFrame>,
    results: &[GwasRowView<'_>],
) -> Result<DataFrame> {
    let mut df = meta
        .context("GWAS metadata missing for output parity")?
        .clone();

    let mut i = Vec::with_capacity(results.len());
    let mut lhs = Vec::with_capacity(results.len());
    let mut op = Vec::with_capacity(results.len());
    let mut rhs = Vec::with_capacity(results.len());
    let mut est = Vec::with_capacity(results.len());
    let mut se = Vec::with_capacity(results.len());
    let mut se_c = Vec::with_capacity(results.len());
    let mut q = Vec::with_capacity(results.len());
    let mut q_snp = Vec::with_capacity(results.len());
    let mut q_snp_df = Vec::with_capacity(results.len());
    let mut q_snp_pval = Vec::with_capacity(results.len());
    let mut fail = Vec::with_capacity(results.len());
    let mut warning = Vec::with_capacity(results.len());
    let mut z_smooth = Vec::with_capacity(results.len());
    let mut has_z = false;
    let mut has_q_snp = false;

    for row in results {
        i.push(row.i as i64);
        lhs.push(row.lhs.to_string());
        op.push(row.op.to_string());
        rhs.push(row.rhs.to_string());
        est.push(row.est);
        se.push(row.se);
        se_c.push(row.se_c);
        q.push(row.q);
        if let Some(val) = row.q_snp {
            q_snp.push(val);
            has_q_snp = true;
        } else {
            q_snp.push(f64::NAN);
        }
        if let Some(val) = row.q_snp_df {
            q_snp_df.push(val);
        } else {
            q_snp_df.push(f64::NAN);
        }
        if let Some(val) = row.q_snp_pval {
            q_snp_pval.push(val);
        } else {
            q_snp_pval.push(f64::NAN);
        }
        fail.push(row.fail.to_string());
        warning.push(row.warning.to_string());
        if let Some(z) = row.z_smooth {
            z_smooth.push(z);
            has_z = true;
        } else {
            z_smooth.push(f64::NAN);
        }
    }

    let mut columns = vec![
        Series::new("i".into(), i),
        Series::new("lhs".into(), lhs),
        Series::new("op".into(), op),
        Series::new("rhs".into(), rhs),
        Series::new("est".into(), est),
        Series::new("se".into(), se),
        Series::new("se_c".into(), se_c),
        Series::new("Q".into(), q),
        Series::new("fail".into(), fail),
        Series::new("warning".into(), warning),
    ];
    if has_q_snp {
        columns.push(Series::new("Q_SNP".into(), q_snp));
        columns.push(Series::new("Q_SNP_df".into(), q_snp_df));
        columns.push(Series::new("Q_SNP_pval".into(), q_snp_pval));
    }
    if has_z {
        columns.push(Series::new("Z_smooth".into(), z_smooth));
    }
    let columns: Vec<Column> = columns.into_iter().map(Column::from).collect();
    df = df.hstack(&columns)?;
    Ok(df)
}

fn reorder_commonfactor_columns(
    df: &mut DataFrame,
    twas: bool,
    smooth_check: bool,
    meta_cols: Vec<&str>,
) -> Result<()> {
    let mut order = Vec::new();
    if !meta_cols.is_empty() {
        order.extend(meta_cols.iter().copied());
    }
    order.extend(["i", "lhs", "op", "rhs", "est", "se_c"]);
    order.extend([
        "Z_Estimate",
        "Pval_Estimate",
        "Q",
        "Q_df",
        "Q_pval",
        "fail",
        "warning",
    ]);
    if smooth_check {
        order.push("Z_smooth");
    }
    let keep: Vec<&str> = order
        .into_iter()
        .filter(|name| df.column(name).is_ok())
        .collect();
    if !keep.is_empty() {
        *df = df.select(keep)?;
    }
    if twas && df.column("rhs").is_ok() {
        let n = df.height();
        df.with_column(Series::new("rhs".into(), vec!["Gene".to_string(); n]).into())?;
    }
    Ok(())
}

pub fn commonfactor_gwas(
    covstruc: &LdscOutput,
    snps: &SumstatsTable,
    config: &CommonFactorGwasConfig,
) -> Result<CommonFactorGwasOutput> {
    let prepared = prepare_commonfactor_gwas(covstruc, snps, config)?;

    let k = prepared.s_ld.len();
    let coords = finite_coords(&prepared.i_ld);
    let run_row = |idx: usize| -> Result<CommonFactorGwasRow> {
        let v_snp = build_v_snp(
            &prepared.se,
            &prepared.i_ld,
            &prepared.var_snp,
            config.gc,
            &coords,
            idx,
        );
        let v_full = build_v_full(k, &prepared.v_ld, prepared.var_snp_se2, &v_snp)?;
        let s_full = build_s_full(k, &prepared.s_ld, &prepared.var_snp, &prepared.beta, idx)?;

        let (v_full, v_smoothed) = smooth_if_needed_full(&v_full)?;
        let (s_full, s_smoothed) = smooth_if_needed_s(&s_full)?;

        let z_smooth = if config.smooth_check {
            let z = if v_smoothed || s_smoothed {
                z_smooth_delta(
                    &prepared.beta,
                    &prepared.se,
                    &prepared.i_ld,
                    config.gc,
                    &s_full,
                    &v_full,
                    idx,
                )
            } else {
                0.0
            };
            Some(z)
        } else {
            None
        };

        let sem_engine = LavaanSemEngine;
        let mut names = Vec::with_capacity(prepared.trait_names.len() + 1);
        let predictor = if config.twas { "Gene" } else { "SNP" };
        names.push(predictor.to_string());
        names.extend(prepared.trait_names.clone());
        let sem = sem_engine.fit(&SemInput {
            s_full: s_full.clone(),
            v_full: v_full.clone(),
            model: commonfactor_model_string(&prepared.trait_names, predictor),
            model_table: None,
            wls_v: None,
            estimation: config.estimation,
            toler: config.toler,
            std_lv: false,
            fix_measurement: false,
            q_snp: true,
            names,
            n_obs: None,
            optim_dx_tol: Some(0.01),
            optim_force_converged: false,
            iter_max: None,
            sample_cov_rescale: false,
        })?;

        Ok(CommonFactorGwasRow {
            i: idx + 1,
            lhs: "F1".to_string(),
            op: "~".to_string(),
            rhs: predictor.to_string(),
            est: sem.est,
            se: sem.se,
            se_c: sem.se_c,
            q: sem.q,
            fail: sem.fail,
            warning: sem.warning,
            z_smooth,
        })
    };

    let rows = if config.parallel {
        let run = || {
            (0..prepared.beta.len())
                .into_par_iter()
                .map(run_row)
                .collect::<Vec<Result<CommonFactorGwasRow>>>()
        };
        let threads = resolve_threads(config.cores, prepared.beta.len());
        let results = run_in_pool(threads, "build gwas thread pool", run)?;
        let mut out = collect_results(results)?;
        out.sort_by_key(|row| row.i);
        out
    } else {
        let mut out = Vec::with_capacity(prepared.beta.len());
        for idx in 0..prepared.beta.len() {
            out.push(run_row(idx)?);
        }
        out
    };

    Ok(CommonFactorGwasOutput {
        prepared,
        results: rows,
    })
}

#[derive(Debug, Clone)]
pub struct UserGwasConfig {
    pub estimation: Estimation,
    pub model: String,
    pub printwarn: bool,
    pub sub: Option<Vec<String>>,
    pub cores: Option<usize>,
    pub toler: Option<f64>,
    pub snp_se: Option<f64>,
    pub parallel: bool,
    pub gc: GenomicControl,
    pub mpi: bool,
    pub smooth_check: bool,
    pub twas: bool,
    pub std_lv: bool,
    pub fix_measurement: bool,
    pub q_snp: bool,
}

pub fn user_gwas(
    covstruc: &LdscOutput,
    snps: &SumstatsTable,
    config: &UserGwasConfig,
) -> Result<UserGwasOutput> {
    let prepared = prepare_user_gwas(covstruc, snps, config)?;
    let k = prepared.s_ld.len();
    let coords = finite_coords(&prepared.i_ld);
    let sub_keys = config.sub.as_ref().map(|subs| {
        subs.iter()
            .map(|s| s.replace(' ', ""))
            .filter(|s| !s.is_empty())
            .collect::<std::collections::HashSet<_>>()
    });

    let run_row = |idx: usize| -> Result<Vec<UserGwasRow>> {
        let v_snp = build_v_snp(
            &prepared.se,
            &prepared.i_ld,
            &prepared.var_snp,
            config.gc,
            &coords,
            idx,
        );
        let v_full = build_v_full(k, &prepared.v_ld, prepared.var_snp_se2, &v_snp)?;
        let s_full = build_s_full(k, &prepared.s_ld, &prepared.var_snp, &prepared.beta, idx)?;

        let (v_full, v_smoothed) = smooth_if_needed_full(&v_full)?;
        let (s_full, s_smoothed) = smooth_if_needed_s(&s_full)?;

        let z_smooth = if config.smooth_check {
            let z = if v_smoothed || s_smoothed {
                z_smooth_delta(
                    &prepared.beta,
                    &prepared.se,
                    &prepared.i_ld,
                    config.gc,
                    &s_full,
                    &v_full,
                    idx,
                )
            } else {
                0.0
            };
            Some(z)
        } else {
            None
        };

        let sem_engine = LavaanSemEngine;
        let mut names = Vec::with_capacity(prepared.trait_names.len() + 1);
        let predictor = if config.twas { "Gene" } else { "SNP" };
        names.push(predictor.to_string());
        names.extend(prepared.trait_names.clone());
        let sem = sem_engine.fit(&SemInput {
            s_full: s_full.clone(),
            v_full: v_full.clone(),
            model: prepared.model.clone(),
            model_table: prepared.model_table.clone(),
            wls_v: None,
            estimation: config.estimation,
            toler: config.toler,
            std_lv: config.std_lv,
            fix_measurement: config.fix_measurement,
            q_snp: config.q_snp,
            names,
            n_obs: None,
            optim_dx_tol: Some(0.01),
            optim_force_converged: false,
            iter_max: None,
            sample_cov_rescale: false,
        })?;

        let q_snp_results = if prepared.q_snp {
            compute_q_snp(
                &prepared.model,
                predictor,
                &prepared.trait_names,
                &v_snp,
                &sem.residual,
            )?
        } else {
            Vec::new()
        };
        if sem.fail != "0" {
            return Ok(vec![UserGwasRow {
                i: idx + 1,
                param_index: 0,
                lhs: String::new(),
                op: String::new(),
                rhs: String::new(),
                est: f64::NAN,
                se: f64::NAN,
                se_c: f64::NAN,
                q: f64::NAN,
                q_snp: None,
                q_snp_df: None,
                q_snp_pval: None,
                fail: sem.fail.clone(),
                warning: sem.warning.clone(),
                z_smooth,
            }]);
        }

        let mut q_snp_map = std::collections::HashMap::new();
        for res in q_snp_results {
            q_snp_map.insert(res.lv, (res.q, res.df, res.pval));
        }

        let mut rows = Vec::new();
        for (param_index, param) in sem.params.iter().enumerate() {
            if param.op == "da" {
                continue;
            }
            if let Some(sub) = &sub_keys {
                let key = format!("{}{}{}", param.lhs, param.op, param.rhs);
                if !sub.contains(&key) {
                    continue;
                }
            }
            let (q_snp, q_snp_df, q_snp_pval) = if prepared.q_snp {
                if let Some((q, df, pval)) = q_snp_map.get(&param.lhs) {
                    (Some(*q), Some(*df), Some(*pval))
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

            rows.push(UserGwasRow {
                i: idx + 1,
                param_index,
                lhs: param.lhs.clone(),
                op: param.op.clone(),
                rhs: param.rhs.clone(),
                est: param.est,
                se: param.se,
                se_c: param.se_c,
                q: sem.q,
                q_snp,
                q_snp_df,
                q_snp_pval,
                fail: sem.fail.clone(),
                warning: sem.warning.clone(),
                z_smooth,
            });
        }

        Ok(rows)
    };

    let rows = if config.parallel {
        let run = || {
            (0..prepared.beta.len())
                .into_par_iter()
                .map(run_row)
                .collect::<Vec<Result<Vec<UserGwasRow>>>>()
        };
        let threads = resolve_threads(config.cores, prepared.beta.len());
        let results = run_in_pool(threads, "build gwas thread pool", run)?;
        let mut out = Vec::new();
        for r in collect_results(results)? {
            out.extend(r);
        }
        out.sort_by_key(|row| (row.i, row.param_index));
        out
    } else {
        let mut out = Vec::new();
        for idx in 0..prepared.beta.len() {
            out.extend(run_row(idx)?);
        }
        out
    };

    Ok(UserGwasOutput {
        prepared,
        results: rows,
    })
}

fn ensure_column(df: &DataFrame, name: &str) -> Result<()> {
    if df.column(name).is_err() {
        return Err(anyhow::anyhow!("Missing required column {name}"));
    }
    Ok(())
}

fn prepare_commonfactor_gwas(
    covstruc: &LdscOutput,
    snps: &SumstatsTable,
    config: &CommonFactorGwasConfig,
) -> Result<CommonFactorPrepared> {
    let s_ld = covstruc.s.clone();
    let v_ld = covstruc.v.clone();
    let mut i_ld = covstruc.i.clone();
    let trait_names = covstruc.trait_names.clone();

    let k = s_ld.len();
    if k == 0 || v_ld.is_empty() {
        return Err(anyhow::anyhow!("LDSC output is empty"));
    }
    if i_ld.len() != k {
        return Err(anyhow::anyhow!(
            "I matrix size {} does not match S matrix size {}",
            i_ld.len(),
            k
        ));
    }
    if trait_names.len() != k {
        return Err(anyhow::anyhow!(
            "trait_names length {} does not match S matrix size {}",
            trait_names.len(),
            k
        ));
    }

    if trait_names.iter().any(|name| name.contains('-')) {
        tracing::warn!("Trait names include '-' which may be misread by downstream lavaan.");
    }

    let z = k * (k + 1) / 2;
    if v_ld.len() != z {
        return Err(anyhow::anyhow!(
            "V matrix must be {z}x{z} for {k} traits; found {}x{}",
            v_ld.len(),
            v_ld.first().map(|r| r.len()).unwrap_or(0)
        ));
    }

    // set univariate intercepts to 1 if estimated below 1
    for (i, row) in i_ld.iter_mut().enumerate().take(k) {
        if row[i].is_finite() && row[i] <= 1.0 {
            row[i] = 1.0;
        }
    }

    let df = &snps.df;
    let beta_cols = find_prefixed_columns(df, "beta.");
    let se_cols = find_prefixed_columns(df, "se.");
    if beta_cols.len() != k || se_cols.len() != k {
        return Err(anyhow::anyhow!(
            "Number of beta/se columns ({}/{}) does not match number of traits {}",
            beta_cols.len(),
            se_cols.len(),
            k
        ));
    }
    let beta = extract_matrix(df, &beta_cols)?;
    let se = extract_matrix(df, &se_cols)?;

    let meta = if config.twas {
        df.select(["HSQ", "Panel", "Gene"])
            .context("select TWAS meta")?
    } else {
        let names = df.get_column_names();
        if names.len() < 6 {
            return Err(anyhow::anyhow!(
                "Expected at least 6 columns in SNP summary stats; found {}",
                names.len()
            ));
        }
        let cols: Vec<&str> = names.iter().take(6).map(|s| s.as_str()).collect();
        df.select(cols).context("select SNP meta")?
    };

    let (snp_ids, a1, a2, maf, var_snp, panel, gene, hsq) = if config.twas {
        ensure_column(df, "Gene")?;
        ensure_column(df, "Panel")?;
        ensure_column(df, "HSQ")?;
        let genes = extract_string_column(df, "Gene")?;
        let panels = extract_string_column(df, "Panel")?;
        let hsq = extract_f64_column(df, "HSQ")?;
        let n = genes.len();
        let empty = vec![String::new(); n];
        let nan = vec![f64::NAN; n];
        (
            genes.clone(),
            empty.clone(),
            empty,
            nan,
            hsq.clone(),
            Some(panels),
            Some(genes),
            Some(hsq),
        )
    } else {
        ensure_column(df, "SNP")?;
        ensure_column(df, "A1")?;
        ensure_column(df, "A2")?;
        ensure_column(df, "MAF")?;
        let snp_ids = extract_string_column(df, "SNP")?;
        let a1 = extract_string_column(df, "A1")?;
        let a2 = extract_string_column(df, "A2")?;
        let maf = extract_f64_column(df, "MAF")?;
        let var_snp: Vec<f64> = maf
            .iter()
            .map(|m| {
                if m.is_finite() {
                    2.0 * m * (1.0 - m)
                } else {
                    f64::NAN
                }
            })
            .collect();
        (snp_ids, a1, a2, maf, var_snp, None, None, None)
    };

    let var_snp_se2 = config.snp_se.unwrap_or(0.0005).powi(2);

    Ok(CommonFactorPrepared {
        trait_names,
        s_ld,
        v_ld,
        i_ld,
        beta,
        se,
        snp_ids,
        a1,
        a2,
        maf,
        var_snp,
        var_snp_se2,
        panel,
        gene,
        hsq,
        twas: config.twas,
        smooth_check: config.smooth_check,
        meta: Some(meta),
    })
}

fn prepare_user_gwas(
    covstruc: &LdscOutput,
    snps: &SumstatsTable,
    config: &UserGwasConfig,
) -> Result<UserGwasPrepared> {
    let prep = prepare_commonfactor_gwas(
        covstruc,
        snps,
        &CommonFactorGwasConfig {
            estimation: config.estimation,
            cores: config.cores,
            toler: config.toler,
            snp_se: config.snp_se,
            parallel: config.parallel,
            gc: config.gc,
            mpi: config.mpi,
            twas: config.twas,
            smooth_check: config.smooth_check,
        },
    )?;

    let predictor = if config.twas { "Gene" } else { "SNP" }.to_string();
    let model_table = if config.fix_measurement {
        let input = FixedMeasurementInput {
            trait_names: &prep.trait_names,
            s_ld: &prep.s_ld,
            v_ld: &prep.v_ld,
            i_ld: &prep.i_ld,
            se: &prep.se,
            var_snp: &prep.var_snp,
            beta: &prep.beta,
            var_snp_se2: prep.var_snp_se2,
            model: &config.model,
            estimation: config.estimation,
            predictor: &predictor,
            gc: config.gc,
            std_lv: config.std_lv,
        };
        Some(build_fixed_measurement_table(input)?)
    } else {
        None
    };

    Ok(UserGwasPrepared {
        trait_names: prep.trait_names,
        s_ld: prep.s_ld,
        v_ld: prep.v_ld,
        i_ld: prep.i_ld,
        beta: prep.beta,
        se: prep.se,
        snp_ids: prep.snp_ids,
        a1: prep.a1,
        a2: prep.a2,
        maf: prep.maf,
        var_snp: prep.var_snp,
        var_snp_se2: prep.var_snp_se2,
        model: config.model.clone(),
        model_table,
        predictor,
        q_snp: config.q_snp,
        panel: prep.panel,
        gene: prep.gene,
        hsq: prep.hsq,
        twas: prep.twas,
        meta: prep.meta,
    })
}

struct FixedMeasurementInput<'a> {
    trait_names: &'a [String],
    s_ld: &'a Matrix,
    v_ld: &'a Matrix,
    i_ld: &'a Matrix,
    se: &'a [Vec<f64>],
    var_snp: &'a [f64],
    beta: &'a [Vec<f64>],
    var_snp_se2: f64,
    model: &'a str,
    estimation: Estimation,
    predictor: &'a str,
    gc: GenomicControl,
    std_lv: bool,
}

fn build_fixed_measurement_table(
    input: FixedMeasurementInput<'_>,
) -> Result<Vec<lavaan_crate::ParTableRow>> {
    let no_snp_model = remove_predictor_lines(input.model, input.predictor);
    let wls_v_ld = if matches!(input.estimation, Estimation::Dwls) {
        Some(build_wls_v_from_gamma(input.v_ld))
    } else {
        None
    };
    let n_obs = match input.estimation {
        Estimation::Dwls => Some(2.0),
        Estimation::Ml => Some(200.0),
    };

    let engine = lavaan_crate::SemEngineImpl;
    let fit_no_snp = engine.fit(&lavaan_crate::SemInput {
        s: input.s_ld.clone(),
        v: input.v_ld.clone(),
        wls_v: wls_v_ld,
        model: no_snp_model,
        model_table: None,
        estimation: map_estimation_lavaan(input.estimation),
        toler: Some(1e-7),
        std_lv: input.std_lv,
        fix_measurement: false,
        q_snp: false,
        names: input.trait_names.to_vec(),
        n_obs,
        optim_dx_tol: Some(0.01),
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    })?;

    let mut table = fit_no_snp.par_table;
    for row in &mut table {
        if row.op == ":=" {
            row.free = 0;
        }
        if row.lhs != row.rhs {
            row.free = 0;
        }
        if !row.ustart.is_finite() {
            row.ustart = row.est;
        }
        if !row.ustart.is_finite() {
            row.ustart = 0.0;
        }
    }

    let predictor_rows = build_predictor_rows(&input)?;
    table.extend(predictor_rows);
    renumber_free_params(&mut table);
    Ok(table)
}

fn build_predictor_rows(
    input: &FixedMeasurementInput<'_>,
) -> Result<Vec<lavaan_crate::ParTableRow>> {
    if input.beta.is_empty() {
        return Ok(Vec::new());
    }
    let v_snp = build_v_snp(
        input.se,
        input.i_ld,
        input.var_snp,
        input.gc,
        &finite_coords(input.i_ld),
        0,
    );
    let v_full = build_v_full(
        input.trait_names.len(),
        input.v_ld,
        input.var_snp_se2,
        &v_snp,
    )?;
    let s_full = build_s_full(
        input.trait_names.len(),
        input.s_ld,
        input.var_snp,
        input.beta,
        0,
    )?;
    let wls_v_full = if matches!(input.estimation, Estimation::Dwls) {
        Some(build_wls_v_from_gamma(&v_full))
    } else {
        None
    };
    let n_obs = match input.estimation {
        Estimation::Dwls => Some(2.0),
        Estimation::Ml => Some(200.0),
    };
    let mut names = Vec::with_capacity(input.trait_names.len() + 1);
    names.push(input.predictor.to_string());
    names.extend_from_slice(input.trait_names);

    let engine = lavaan_crate::SemEngineImpl;
    let fit_full = engine.fit(&lavaan_crate::SemInput {
        s: s_full,
        v: v_full,
        wls_v: wls_v_full,
        model: input.model.to_string(),
        model_table: None,
        estimation: map_estimation_lavaan(input.estimation),
        toler: Some(1e-7),
        std_lv: input.std_lv,
        fix_measurement: false,
        q_snp: false,
        names,
        n_obs,
        optim_dx_tol: Some(0.01),
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    })?;

    let mut out = Vec::new();
    for mut row in fit_full.par_table {
        if row.lhs == input.predictor || row.rhs == input.predictor || row.op == ":=" {
            if !row.ustart.is_finite() {
                row.ustart = row.est;
            }
            if !row.ustart.is_finite() {
                row.ustart = 0.0;
            }
            out.push(row);
        }
    }
    Ok(out)
}

fn remove_predictor_lines(model: &str, predictor: &str) -> String {
    let mut out = Vec::new();
    for line in model.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.contains(":=") {
            continue;
        }
        if trimmed.contains(predictor) {
            continue;
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

fn build_wls_v_from_gamma(gamma: &Matrix) -> Matrix {
    let n = gamma.len();
    let mut out = vec![vec![0.0; n]; n];
    for (i, row_out) in out.iter_mut().enumerate() {
        if let Some(row) = gamma.get(i) {
            let val = row.get(i).copied().unwrap_or(f64::NAN);
            if val.is_finite() && val > 0.0 {
                row_out[i] = 1.0 / val;
            }
        }
    }
    out
}

fn renumber_free_params(rows: &mut [lavaan_crate::ParTableRow]) {
    let mut label_map = std::collections::HashMap::new();
    let mut next = 1;
    for row in rows.iter_mut() {
        if row.free == 0 {
            continue;
        }
        if let Some(label) = &row.label {
            if let Some(idx) = label_map.get(label) {
                row.free = *idx;
            } else {
                let idx = next;
                next += 1;
                label_map.insert(label.clone(), idx);
                row.free = idx;
            }
        } else {
            row.free = next;
            next += 1;
        }
    }
}

fn map_estimation_lavaan(est: Estimation) -> lavaan_crate::Estimation {
    match est {
        Estimation::Dwls => lavaan_crate::Estimation::Dwls,
        Estimation::Ml => lavaan_crate::Estimation::Ml,
    }
}

#[derive(Debug, Clone)]
struct QSnpResult {
    lv: String,
    q: f64,
    df: f64,
    pval: f64,
}

fn compute_q_snp(
    model: &str,
    predictor: &str,
    trait_names: &[String],
    v_snp: &Matrix,
    residual: &Matrix,
) -> Result<Vec<QSnpResult>> {
    let spec = match lavaan_crate::parser::parse_model(model) {
        Ok(spec) => spec,
        Err(_) => return Ok(Vec::new()),
    };

    let mut latent_set = std::collections::HashSet::new();
    for line in &spec.lines {
        if let ModelOp::Measure = line.op {
            latent_set.insert(line.lhs.clone());
        }
    }

    let mut lv_with_predictor = Vec::new();
    let mut lv_seen = std::collections::HashSet::new();
    for line in &spec.lines {
        if let ModelOp::Regress = line.op
            && line.terms.iter().any(|t| t.var == predictor)
            && lv_seen.insert(line.lhs.clone())
        {
            lv_with_predictor.push(line.lhs.clone());
        }
    }

    let mut indicators_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for line in &spec.lines {
        if let ModelOp::Measure = line.op {
            let entry = indicators_map.entry(line.lhs.clone()).or_default();
            for term in &line.terms {
                if term.var != "1" {
                    entry.push(term.var.clone());
                }
            }
        }
    }

    let mut results = Vec::new();
    for lv in lv_with_predictor {
        if !latent_set.contains(&lv) {
            continue;
        }
        let indicators = indicators_map.get(&lv).cloned().unwrap_or_default();
        let (q, df, pval) = q_snp_for_indicators(&indicators, trait_names, v_snp, residual)?;
        results.push(QSnpResult { lv, q, df, pval });
    }
    Ok(results)
}

fn q_snp_for_indicators(
    indicators: &[String],
    trait_names: &[String],
    v_snp: &Matrix,
    residual: &Matrix,
) -> Result<(f64, f64, f64)> {
    if indicators.is_empty() {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }
    let mut idx = Vec::with_capacity(indicators.len());
    for name in indicators {
        if let Some(pos) = trait_names.iter().position(|t| t == name) {
            idx.push(pos);
        } else {
            let df = indicators.len() as f64 - 1.0;
            return Ok((f64::NAN, df, f64::NAN));
        }
    }

    let k = idx.len();
    let mut v_sub = Array2::<f64>::zeros((k, k));
    for (i, &ri) in idx.iter().enumerate() {
        for (j, &cj) in idx.iter().enumerate() {
            v_sub[(i, j)] = v_snp[ri][cj];
        }
    }

    let (eigvals, eigvecs) = v_sub.eigh(UPLO::Lower)?;
    let mut inv_vals = eigvals.to_vec();
    for v in &mut inv_vals {
        *v = 1.0 / *v;
    }
    let inv_diag = Array2::from_diag(&ndarray::Array1::from_vec(inv_vals));
    let middle = eigvecs.dot(&inv_diag).dot(&eigvecs.t());

    let mut eta = Array2::<f64>::zeros((k, 1));
    for (i, &ri) in idx.iter().enumerate() {
        eta[(i, 0)] = residual[0][1 + ri];
    }
    let q_mat = eta.t().dot(&middle).dot(&eta);
    let q = q_mat[(0, 0)];
    let df = k as f64 - 1.0;
    let pval = if df > 0.0 && q.is_finite() {
        let chi = ChiSquared::new(df).unwrap();
        1.0 - chi.cdf(q)
    } else {
        f64::NAN
    };
    Ok((q, df, pval))
}

fn all_coords(k: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            coords.push((i, j));
        }
    }
    coords
}

fn finite_coords(i_ld: &Matrix) -> Vec<(usize, usize)> {
    let k = i_ld.len();
    let mut coords = Vec::new();
    for (i, row) in i_ld.iter().enumerate().take(k) {
        for (j, value) in row.iter().enumerate().take(k) {
            if value.is_finite() {
                coords.push((i, j));
            }
        }
    }
    if coords.is_empty() {
        return all_coords(k);
    }
    coords
}

fn build_v_snp(
    se: &[Vec<f64>],
    i_ld: &Matrix,
    var_snp: &[f64],
    gc: GenomicControl,
    coords: &[(usize, usize)],
    idx: usize,
) -> Matrix {
    let k = i_ld.len();
    let mut v = vec![vec![0.0; k]; k];
    for &(x, y) in coords {
        let se_x = se[idx][x];
        let se_y = se[idx][y];
        let var = var_snp[idx];
        let i_xy = i_ld[x][y];
        let i_xx = i_ld[x][x];
        let i_yy = i_ld[y][y];
        let value = if x != y {
            match gc {
                GenomicControl::Conserv => se_y * se_x * i_xy * i_xx * i_yy * var * var,
                GenomicControl::Standard => {
                    se_y * se_x * i_xy * i_xx.sqrt() * i_yy.sqrt() * var * var
                }
                GenomicControl::None => se_y * se_x * i_xy * var * var,
            }
        } else {
            match gc {
                GenomicControl::Conserv => (se_x * i_xx * var).powi(2),
                GenomicControl::Standard => (se_x * i_xx.sqrt() * var).powi(2),
                GenomicControl::None => (se_x * var).powi(2),
            }
        };
        v[x][y] = value;
    }
    v
}

fn build_v_full(k: usize, v_ld: &Matrix, var_snp_se2: f64, v_snp: &Matrix) -> Result<Matrix> {
    let z = (k + 1) * (k + 2) / 2;
    let mut v_full = vec![vec![0.0; z]; z];
    for (i, row) in v_full.iter_mut().enumerate().take(z) {
        row[i] = 1.0;
    }
    let start = k + 1;
    for i in 0..v_ld.len() {
        for j in 0..v_ld[i].len() {
            v_full[start + i][start + j] = v_ld[i][j];
        }
    }
    v_full[0][0] = var_snp_se2;
    for i in 0..k {
        for j in 0..k {
            v_full[1 + i][1 + j] = v_snp[i][j];
        }
    }
    Ok(v_full)
}

fn build_s_full(
    k: usize,
    s_ld: &Matrix,
    var_snp: &[f64],
    beta: &[Vec<f64>],
    idx: usize,
) -> Result<Matrix> {
    let mut s_full = vec![vec![0.0; k + 1]; k + 1];
    for (i, row) in s_full.iter_mut().enumerate().take(k + 1) {
        row[i] = 1.0;
    }
    let mut s_snp = vec![0.0; k + 1];
    s_snp[0] = var_snp[idx];
    for p in 0..k {
        s_snp[p + 1] = var_snp[idx] * beta[idx][p];
    }
    for i in 0..k {
        for j in 0..k {
            s_full[i + 1][j + 1] = s_ld[i][j];
        }
    }
    for i in 0..k + 1 {
        s_full[i][0] = s_snp[i];
        s_full[0][i] = s_snp[i];
    }
    Ok(s_full)
}

fn smooth_if_needed_full(matrix: &Matrix) -> Result<(Matrix, bool)> {
    // TODO(Matrix::nearPD): Replace eigenvalue clipping with a true nearPD algorithm if available.
    let (smoothed, was_smoothed, _) = crate::sem::smooth_if_needed(matrix)?;
    Ok((smoothed, was_smoothed))
}

fn smooth_if_needed_s(matrix: &Matrix) -> Result<(Matrix, bool)> {
    // TODO(Matrix::nearPD): Replace eigenvalue clipping with a true nearPD algorithm if available.
    let (smoothed, was_smoothed, _) = crate::sem::smooth_if_needed(matrix)?;
    Ok((smoothed, was_smoothed))
}

fn z_smooth_delta(
    beta: &[Vec<f64>],
    se: &[Vec<f64>],
    i_ld: &Matrix,
    gc: GenomicControl,
    s_full: &Matrix,
    v_full: &Matrix,
    idx: usize,
) -> f64 {
    let z_pre = get_z_pre(beta, se, i_ld, gc, idx);
    let se_smooth = se_from_v_full(v_full);
    let ks = s_full.len();
    let mut max_diff = 0.0;
    for t in 1..ks {
        let z = if se_smooth[t][t] != 0.0 {
            s_full[t][0] / se_smooth[t][t]
        } else {
            0.0
        };
        let diff = (z - z_pre[t - 1]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

fn se_from_v_full(v_full: &Matrix) -> Matrix {
    let n = v_full.len();
    let mut se = vec![vec![0.0; n]; n];
    for i in 0..n {
        if v_full[i][i].is_finite() && v_full[i][i] >= 0.0 {
            se[i][i] = v_full[i][i].sqrt();
        }
    }
    se
}

fn get_z_pre(
    beta: &[Vec<f64>],
    se: &[Vec<f64>],
    i_ld: &Matrix,
    gc: GenomicControl,
    idx: usize,
) -> Vec<f64> {
    let k = i_ld.len();
    let mut out = vec![0.0; k];
    for p in 0..k {
        let denom = match gc {
            GenomicControl::Conserv => se[idx][p] * i_ld[p][p],
            GenomicControl::Standard => se[idx][p] * i_ld[p][p].sqrt(),
            GenomicControl::None => se[idx][p],
        };
        out[p] = if denom != 0.0 {
            beta[idx][p] / denom
        } else {
            0.0
        };
    }
    out
}

fn commonfactor_model_string(trait_names: &[String], predictor: &str) -> String {
    if trait_names.is_empty() {
        return format!("F1 =~ {predictor}\nF1 ~ {predictor}");
    }
    let mut model = format!("F1 =~ {}", trait_names[0]);
    for name in trait_names.iter().skip(1) {
        model.push_str(" + ");
        model.push_str(name);
    }
    model.push_str(&format!("\nF1 ~ {predictor}"));
    for name in trait_names {
        model.push_str(&format!("\n{name} ~ 0*{predictor}"));
    }
    model
}

fn find_prefixed_columns(df: &DataFrame, prefix: &str) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter(|name| name.starts_with(prefix))
        .map(|s| s.to_string())
        .collect()
}

fn extract_f64_column(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let series = df
        .column(name)
        .with_context(|| format!("column {name}"))?
        .f64()
        .context("cast f64")?;
    Ok(series.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn extract_string_column(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let series = df
        .column(name)
        .with_context(|| format!("column {name}"))?
        .str()
        .context("cast string")?;
    Ok(series
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect())
}

fn extract_matrix(df: &DataFrame, cols: &[String]) -> Result<Vec<Vec<f64>>> {
    let n = df.height();
    let mut out = vec![vec![f64::NAN; cols.len()]; n];
    for (j, name) in cols.iter().enumerate() {
        let series = df
            .column(name)
            .with_context(|| format!("column {name}"))?
            .f64()
            .context("cast f64")?;
        for (i, v) in series.into_iter().enumerate() {
            out[i][j] = v.unwrap_or(f64::NAN);
        }
    }
    Ok(out)
}
