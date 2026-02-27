use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use polars::prelude::*;
use rayon::prelude::*;

use crate::df_utils::{
    add_z_score, ensure_f64, ensure_utf8, filter_allele_mismatch, filter_missing, filter_non_acgt,
    flip_effect_if_needed, uppercase_alleles,
};
use crate::io::read_table;
use crate::logging::log_line;
use crate::parallel::{collect_results, run_in_pool};
use crate::schema::{ColumnMapConfig, resolve_column_map};
use crate::types::SumstatsTable;

#[derive(Debug, Clone)]
pub struct SumstatsConfig {
    pub files: Vec<PathBuf>,
    pub reference: PathBuf,
    pub trait_names: Option<Vec<String>>,
    pub se_logit: Vec<bool>,
    pub ols: Option<Vec<bool>>,
    pub linprob: Option<Vec<bool>>,
    pub n: Option<Vec<f64>>,
    pub betas: Option<Vec<String>>,
    pub info_filter: f64,
    pub maf_filter: f64,
    pub keep_indel: bool,
    pub parallel: bool,
    pub cores: Option<usize>,
    pub ambig: bool,
    pub direct_filter: bool,
    pub column_names: HashMap<String, String>,
    pub log_name: Option<String>,
}

pub fn sumstats(config: &SumstatsConfig) -> Result<SumstatsTable> {
    validate_lengths(config)?;
    let mut log = open_log_file(config)?;
    log_line(
        &mut log,
        &format!("Preparing {} summary statistics", config.files.len()),
        true,
    )?;

    let mut ref_df = read_table(&config.reference)?;
    if config.ambig {
        ref_df = remove_strand_ambig(ref_df)?;
    }
    if ref_df.column("MAF").is_ok() {
        let maf = ref_df.column("MAF")?.as_series().context("MAF")?.f64()?;
        let mask = maf.gt_eq(config.maf_filter);
        ref_df = ref_df.filter(&mask)?;
    }
    let ref_work = build_ref_work(&ref_df)?;

    let output_frames = if config.parallel {
        log_line(
            &mut log,
            "As parallel sumstats was requested, logs of each file will be saved separately.",
            true,
        )?;
        run_parallel_sumstats(config, &ref_df, &ref_work)?
    } else {
        run_sequential_sumstats(config, &ref_df, &ref_work, &mut log)?
    };

    let mut merged = ref_df.clone();
    for frame in output_frames {
        merged = merged.join(&frame, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    }

    write_sumstats_output(&merged, config)?;
    Ok(SumstatsTable { df: merged })
}

fn run_sequential_sumstats(
    config: &SumstatsConfig,
    ref_df: &DataFrame,
    ref_work: &DataFrame,
    log: &mut File,
) -> Result<Vec<DataFrame>> {
    let mut output_frames = Vec::new();
    for (idx, file) in config.files.iter().enumerate() {
        let df = run_sumstats_for_file(config, idx, file, ref_df, ref_work, log)?;
        output_frames.push(df);
    }
    Ok(output_frames)
}

fn run_parallel_sumstats(
    config: &SumstatsConfig,
    ref_df: &DataFrame,
    ref_work: &DataFrame,
) -> Result<Vec<DataFrame>> {
    let files: Vec<(usize, PathBuf)> = config
        .files
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.clone()))
        .collect();

    let run = || {
        files
            .par_iter()
            .map(|(idx, file)| {
                let mut log = open_trait_log(config, *idx, file)?;
                let df = run_sumstats_for_file(config, *idx, file, ref_df, ref_work, &mut log)?;
                Ok((*idx, df))
            })
            .collect::<Vec<Result<(usize, DataFrame)>>>()
    };

    let results = run_in_pool(config.cores, "build sumstats thread pool", run)?;

    let mut collected = collect_results(results)?;
    collected.sort_by_key(|(idx, _)| *idx);
    Ok(collected.into_iter().map(|(_, df)| df).collect())
}

fn run_sumstats_for_file(
    config: &SumstatsConfig,
    idx: usize,
    file: &Path,
    _ref_df: &DataFrame,
    ref_work: &DataFrame,
    log: &mut File,
) -> Result<DataFrame> {
    let trait_name = resolve_trait_name(config, idx, file)?;
    log_line(
        log,
        &format!("Preparing summary statistics for {}", file.display()),
        true,
    )?;

    let mut df = read_table(file)?;
    df = handle_user_columns(df, &config.column_names, file, config, idx, log)?;
    df = remove_duplicate_snps(df, log, file)?;

    if config.direct_filter {
        df = filter_direction(df, log, file)?;
    }

    df = ensure_utf8(df, &["SNP", "A1", "A2"])?;
    df = ensure_f64(df, &["P", "EFFECT", "SE", "N", "INFO", "MAF"])?;
    df = uppercase_alleles(df)?;
    if !config.keep_indel {
        let (df_tmp, removed) = filter_non_acgt(df, "A1", "A2")?;
        df = df_tmp;
        if removed > 0 {
            log_line(
                log,
                &format!(
                    "{removed} row(s) removed from {} due to non-ACGT alleles",
                    file.display()
                ),
                true,
            )?;
        }
    }

    let before = df.height();
    df = df.join(ref_work, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!("{removed} rows removed after merging with reference"),
            true,
        )?;
    }

    let (df_tmp, removed) = filter_missing(df, "P")?;
    df = df_tmp;
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} rows were removed from {} due to missing values in P",
                file.display()
            ),
            true,
        )?;
    }
    let (df_tmp, removed) = filter_missing(df, "EFFECT")?;
    df = df_tmp;
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} rows were removed from {} due to missing values in EFFECT",
                file.display()
            ),
            true,
        )?;
    }

    df = apply_maf_from_sumstats(df, log, file)?;
    df = detect_or_log_transform(df, log)?;
    df = remove_effect_zero(df, log, file)?;
    df = add_z_score(df, "EFFECT", "P")?;
    df = flip_effect_if_needed(df, "A1", "A1_REF", "A2_REF", "EFFECT")?;
    let (df_tmp, removed) = filter_allele_mismatch(df, "A1", "A2", "A1_REF", "A2_REF")?;
    df = df_tmp;
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} row(s) removed from {} due to allele mismatch",
                file.display()
            ),
            true,
        )?;
    }
    df = p_value_sanity_check(df, log, file)?;
    df = filter_info(df, config.info_filter, log)?;

    let se_logit = *config.se_logit.get(idx).unwrap_or(&false);
    let linprob = config
        .linprob
        .as_ref()
        .and_then(|v| v.get(idx))
        .copied()
        .unwrap_or(false);
    let ols = config
        .ols
        .as_ref()
        .and_then(|v| v.get(idx))
        .copied()
        .unwrap_or(false);
    let beta_col = config.betas.as_ref().and_then(|v| v.get(idx)).cloned();
    let n_val = config
        .n
        .as_ref()
        .and_then(|v| v.get(idx))
        .copied()
        .filter(|v| v.is_finite());

    if ols && beta_col.is_some() {
        log_line(
            log,
            &format!(
                "OLS: using standardized betas for {} (effect column assumed standardized).",
                file.display()
            ),
            true,
        )?;
    }
    if linprob {
        log_line(
            log,
            &format!(
                "Linprob: applying transformation to back out logistic betas for {}.",
                file.display()
            ),
            true,
        )?;
    }

    if let Some(n_val) = n_val {
        if ols {
            log_line(
                log,
                &format!(
                    "Using user provided N of {n_val} for {} (OLS).",
                    file.display()
                ),
                true,
            )?;
        } else if linprob {
            log_line(
                log,
                &format!(
                    "Using user provided N of {n_val} for {} (linprob).",
                    file.display()
                ),
                true,
            )?;
        } else {
            log_line(
                log,
                &format!("Using user provided N of {n_val} for {}.", file.display()),
                true,
            )?;
        }
    }

    df = apply_standardization(df, se_logit, linprob, ols, beta_col, n_val, log, file)?;
    df = filter_output_rows(df, linprob)?;

    log_line(
        log,
        &format!(
            "{} SNPs are left in the summary statistics file {} after QC and merging with the reference file.",
            df.height(),
            file.display()
        ),
        true,
    )?;

    let avg_z = average_z(&df)?;
    if avg_z > 5.0 {
        log_line(
            log,
            &format!("WARNING: mean |Z| > 5 for {trait_name}. Check column mappings."),
            true,
        )?;
    }

    let beta_name = if let Some(names) = &config.trait_names {
        format!("beta.{}", names[idx])
    } else {
        format!("beta.{}", idx + 1)
    };
    let se_name = if let Some(names) = &config.trait_names {
        format!("se.{}", names[idx])
    } else {
        format!("se.{}", idx + 1)
    };

    let mut df = df.select(["SNP", "BETA_STD", "SE_STD"])?;
    df.rename("BETA_STD", beta_name.into())?;
    df.rename("SE_STD", se_name.into())?;
    Ok(df)
}

fn resolve_trait_name(config: &SumstatsConfig, idx: usize, file: &Path) -> Result<String> {
    if let Some(names) = &config.trait_names {
        return Ok(names[idx].clone());
    }
    Ok(file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("trait")
        .to_string())
}

fn open_log_file(config: &SumstatsConfig) -> Result<File> {
    let log_name = if let Some(name) = &config.log_name {
        name.clone()
    } else if let Some(names) = &config.trait_names {
        let mut joined = names.join("_");
        joined = joined.replace('/', "");
        if joined.len() > 200 {
            joined.truncate(100);
        }
        if joined.is_empty() {
            "sumstats".to_string()
        } else {
            joined
        }
    } else {
        "sumstats".to_string()
    };
    let path = format!("{log_name}_sumstats.log");
    Ok(File::create(path)?)
}

fn open_trait_log(config: &SumstatsConfig, idx: usize, file: &Path) -> Result<File> {
    let mut name = resolve_trait_name(config, idx, file)?;
    name = name.replace('/', "");
    if name.is_empty() {
        name = format!("trait_{}", idx + 1);
    }
    let path = format!("{name}_sumstats.log");
    Ok(File::create(path)?)
}

fn build_ref_work(ref_df: &DataFrame) -> Result<DataFrame> {
    let mut cols = Vec::new();
    if ref_df.column("SNP").is_ok() {
        cols.push("SNP");
    }
    if ref_df.column("A1").is_ok() {
        cols.push("A1");
    }
    if ref_df.column("A2").is_ok() {
        cols.push("A2");
    }
    if ref_df.column("MAF").is_ok() {
        cols.push("MAF");
    }

    let mut df = if cols.is_empty() {
        ref_df.clone()
    } else {
        ref_df.select(cols)?
    };

    if df.column("A1").is_ok() {
        df.rename("A1", "A1_REF".into())?;
    }
    if df.column("A2").is_ok() {
        df.rename("A2", "A2_REF".into())?;
    }
    if df.column("MAF").is_ok() {
        df.rename("MAF", "MAF_REF".into())?;
    }
    Ok(df)
}

fn remove_strand_ambig(mut df: DataFrame) -> Result<DataFrame> {
    let (a1_name, a2_name) = if df.column("A1_REF").is_ok() && df.column("A2_REF").is_ok() {
        ("A1_REF", "A2_REF")
    } else if df.column("A1").is_ok() && df.column("A2").is_ok() {
        ("A1", "A2")
    } else {
        return Ok(df);
    };
    let a1 = df.column(a1_name)?.as_series().context("A1")?.str()?;
    let a2 = df.column(a2_name)?.as_series().context("A2")?.str()?;
    let mask: BooleanChunked = a1
        .into_iter()
        .zip(a2)
        .map(|(a1v, a2v)| {
            !matches!(
                (a1v, a2v),
                (Some("T"), Some("A"))
                    | (Some("A"), Some("T"))
                    | (Some("C"), Some("G"))
                    | (Some("G"), Some("C"))
            )
        })
        .collect();
    df = df.filter(&mask)?;
    Ok(df)
}

fn handle_user_columns(
    mut df: DataFrame,
    column_names: &HashMap<String, String>,
    file: &Path,
    config: &SumstatsConfig,
    idx: usize,
    log: &mut File,
) -> Result<DataFrame> {
    let headers: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let se_logit = *config.se_logit.get(idx).unwrap_or(&false);
    let linprob = config
        .linprob
        .as_ref()
        .and_then(|v| v.get(idx))
        .copied()
        .unwrap_or(false);
    let ols = config
        .ols
        .as_ref()
        .and_then(|v| v.get(idx))
        .copied()
        .unwrap_or(false);

    let stop_on_missing = if linprob {
        vec!["EFFECT".into(), "SNP".into()]
    } else if se_logit || (!linprob && !ols) {
        vec!["EFFECT".into(), "SNP".into(), "SE".into()]
    } else {
        vec!["EFFECT".into(), "SNP".into()]
    };

    let map = resolve_column_map(
        &headers,
        &ColumnMapConfig {
            userprovided: column_names.clone(),
            check_single: vec![
                "P".into(),
                "A1".into(),
                "A2".into(),
                "EFFECT".into(),
                "SNP".into(),
            ],
            warn_for_missing: vec!["P".into(), "A1".into(), "A2".into(), "N".into()],
            stop_on_missing,
            warn_z_as_effect: (!linprob && !ols),
            n_provided: config.n.as_ref().and_then(|v| v.get(idx)).is_some(),
            filename: Some(file.display().to_string()),
        },
    )?;

    for msg in &map.info {
        log_line(log, msg, true)?;
    }
    for msg in &map.warnings {
        log_line(log, msg, true)?;
    }

    df.set_column_names(&map.headers)?;

    if let Some(n_val) = config.n.as_ref().and_then(|v| v.get(idx)).copied() {
        let n_series = Series::new("N".into(), vec![n_val; df.height()]);
        df.with_column(n_series.into())?;
    }
    Ok(df)
}

fn remove_duplicate_snps(mut df: DataFrame, log: &mut File, file: &Path) -> Result<DataFrame> {
    if df.column("SNP").is_err() {
        return Ok(df);
    }
    let before = df.height();
    let counts = df
        .clone()
        .lazy()
        .group_by([col("SNP")])
        .agg([col("SNP").count().alias("dup_count")])
        .collect()?;
    df = df.join(&counts, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let dup = df
        .column("dup_count")?
        .as_series()
        .context("dup_count")?
        .u32()?;
    let mask = dup.equal(1);
    df = df.filter(&mask)?;
    df.drop_in_place("dup_count")?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} rows removed from {} due to duplicated SNP entries",
                file.display()
            ),
            true,
        )?;
    }
    Ok(df)
}

fn filter_direction(mut df: DataFrame, log: &mut File, file: &Path) -> Result<DataFrame> {
    if df.column("DIRECTION").is_err() {
        log_line(
            log,
            "No DIRECTION column, cannot filter on missingness",
            true,
        )?;
        return Ok(df);
    }
    let before = df.height();
    let direction = df
        .column("DIRECTION")?
        .as_series()
        .context("DIRECTION")?
        .str()?;
    let mask: BooleanChunked = direction
        .into_iter()
        .map(|v| match v {
            Some(s) => {
                let total = s.len().max(1) as f64;
                let missing = s.chars().filter(|c| *c == '?').count() as f64;
                (missing / total) < 0.5
            }
            None => false,
        })
        .collect();
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} rows removed from {} due to missingness in DIRECTION",
                file.display()
            ),
            true,
        )?;
    }
    Ok(df)
}

fn apply_maf_from_sumstats(mut df: DataFrame, log: &mut File, file: &Path) -> Result<DataFrame> {
    if df.column("MAF").is_ok() {
        let maf = df.column("MAF")?.as_series().context("MAF")?.f64()?;
        let maf = maf
            .apply(|v| v.map(|x| if x > 0.5 { 1.0 - x } else { x }))
            .into_series();
        let mut s = maf;
        s.rename("MAF".into());
        df.with_column(s.into())?;

        let before = df.height();
        let maf = df.column("MAF")?.as_series().context("MAF")?.f64()?;
        let mask = maf.lt(1.0) & maf.gt(0.0);
        df = df.filter(&mask)?;
        let removed = before.saturating_sub(df.height());
        if removed > 0 {
            log_line(
                log,
                &format!(
                    "{removed} rows removed due to MAF exactly 0 or 1 in {}",
                    file.display()
                ),
                true,
            )?;
        }

        let maf = df.column("MAF")?.as_series().context("MAF")?.f64()?;
        let var = maf.apply(|v| v.map(|x| 2.0 * x * (1.0 - x))).into_series();
        let mut v = var;
        v.rename("VAR_SNP".into());
        df.with_column(v.into())?;
    } else if df.column("MAF_REF").is_ok() {
        let maf = df
            .column("MAF_REF")?
            .as_series()
            .context("MAF_REF")?
            .f64()?;
        let var = maf.apply(|v| v.map(|x| 2.0 * x * (1.0 - x))).into_series();
        let mut v = var;
        v.rename("VAR_SNP".into());
        df.with_column(v.into())?;
    }
    Ok(df)
}

fn detect_or_log_transform(mut df: DataFrame, log: &mut File) -> Result<DataFrame> {
    if df.column("EFFECT").is_err() {
        return Ok(df);
    }
    let effect = df.column("EFFECT")?.as_series().context("EFFECT")?.f64()?;
    let median = effect.median().unwrap_or(0.0);
    if median.round() == 1.0 {
        let logged = effect.apply(|v| v.map(|x| x.ln())).into_series();
        let mut series = logged;
        series.rename("EFFECT".into());
        df.with_column(series.into())?;
        log_line(
            log,
            "Effect column interpreted as OR; log transformed.",
            true,
        )?;
    } else {
        log_line(log, "Effect column interpreted as beta.", true)?;
    }
    Ok(df)
}

fn remove_effect_zero(mut df: DataFrame, log: &mut File, file: &Path) -> Result<DataFrame> {
    if df.column("EFFECT").is_err() {
        return Ok(df);
    }
    let before = df.height();
    let effect = df.column("EFFECT")?.as_series().context("EFFECT")?.f64()?;
    let mask = effect.not_equal(0.0);
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!(
                "{removed} rows removed from {} due to EFFECT=0",
                file.display()
            ),
            true,
        )?;
    }
    Ok(df)
}

fn p_value_sanity_check(df: DataFrame, log: &mut File, file: &Path) -> Result<DataFrame> {
    if df.column("P").is_err() {
        return Ok(df);
    }
    let p = df.column("P")?.as_series().context("P")?.f64()?;
    let mut bad = 0usize;
    for v in p.into_iter().flatten() {
        if !(0.0..=1.0).contains(&v) {
            bad += 1;
        }
    }
    if bad > 100 {
        log_line(
            log,
            &format!(
                "In excess of 100 SNPs have P val above 1 or below 0 in {}",
                file.display()
            ),
            true,
        )?;
    }
    Ok(df)
}

fn filter_info(mut df: DataFrame, info_filter: f64, log: &mut File) -> Result<DataFrame> {
    if df.column("INFO").is_err() {
        log_line(log, "No INFO column, cannot filter on INFO", true)?;
        return Ok(df);
    }
    let before = df.height();
    let info = df.column("INFO")?.as_series().context("INFO")?.f64()?;
    let mask = info.gt_eq(info_filter);
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!("{removed} rows removed due to INFO below {info_filter}"),
            true,
        )?;
    }
    Ok(df)
}

fn filter_output_rows(mut df: DataFrame, linprob: bool) -> Result<DataFrame> {
    if df.column("BETA_STD").is_err() || df.column("SE_STD").is_err() {
        return Ok(df);
    }
    let beta = df
        .column("BETA_STD")?
        .as_series()
        .context("BETA_STD")?
        .f64()?;
    let se = df.column("SE_STD")?.as_series().context("SE_STD")?.f64()?;

    let mask: BooleanChunked = beta
        .into_iter()
        .zip(se)
        .map(|(b, s)| match (b, s) {
            (Some(bv), Some(sv)) if !bv.is_nan() && !sv.is_nan() => {
                if linprob {
                    bv != 0.0 && sv != 0.0
                } else {
                    true
                }
            }
            _ => false,
        })
        .collect();

    df = df.filter(&mask)?;
    Ok(df)
}

#[allow(clippy::too_many_arguments)]
fn apply_standardization(
    mut df: DataFrame,
    se_logit: bool,
    linprob: bool,
    ols: bool,
    beta_col: Option<String>,
    n_val: Option<f64>,
    log: &mut File,
    file: &Path,
) -> Result<DataFrame> {
    let var = df
        .column("VAR_SNP")?
        .as_series()
        .context("VAR_SNP")?
        .f64()?;
    let effect = df.column("EFFECT")?.as_series().context("EFFECT")?.f64()?;
    let z = df.column("Z")?.as_series().context("Z")?.f64()?;
    let se = df
        .column("SE")
        .ok()
        .and_then(|c| c.as_series())
        .and_then(|s| s.f64().ok());

    let n_series = if let Some(n_val) = n_val {
        Float64Chunked::from_vec("N".into(), vec![n_val; df.height()])
    } else if let Ok(n) = df.column("N") {
        n.as_series().context("N")?.f64()?.clone()
    } else {
        Float64Chunked::from_vec("N".into(), vec![f64::NAN; df.height()])
    };

    let n_missing = n_series
        .into_iter()
        .all(|v| v.map(|x| x.is_nan()).unwrap_or(true));

    if (ols || linprob) && n_missing {
        log_line(
            log,
            &format!(
                "ERROR: Sample size (N) is required for {} but was missing",
                file.display()
            ),
            true,
        )?;
    }

    let mut beta_std = Vec::with_capacity(df.height());
    let mut se_std = Vec::with_capacity(df.height());

    let se_vec: Option<Vec<Option<f64>>> = se.as_ref().map(|s| s.into_iter().collect());

    for i in 0..df.height() {
        let var_i = var.get(i);
        let eff_i = effect.get(i);
        let z_i = z.get(i);
        let n_i = n_series.get(i);
        let se_i = se_vec.as_ref().and_then(|v| v.get(i)).copied().flatten();

        let (beta_val, se_val) = if ols {
            let effect_val = if beta_col.is_some() {
                eff_i
            } else if let (Some(zv), Some(nv), Some(vv)) = (z_i, n_i, var_i) {
                let denom = (nv * vv).sqrt();
                Some(if denom == 0.0 { f64::NAN } else { zv / denom })
            } else {
                None
            };

            let se_val = match (effect_val, z_i) {
                (Some(ev), Some(zv)) if zv != 0.0 => Some((ev / zv).abs()),
                _ => None,
            };
            (effect_val, se_val)
        } else if linprob {
            if let (Some(zv), Some(nv), Some(vv)) = (z_i, n_i, var_i) {
                let denom_lp = ((nv / 4.0) * vv).sqrt();
                let effect_lp = if denom_lp == 0.0 {
                    f64::NAN
                } else {
                    zv / denom_lp
                };
                let se_lp = if denom_lp == 0.0 {
                    f64::NAN
                } else {
                    1.0 / denom_lp
                };

                let denom =
                    (effect_lp * effect_lp * vv + (std::f64::consts::PI.powi(2) / 3.0)).sqrt();
                let b = if denom == 0.0 {
                    f64::NAN
                } else {
                    effect_lp / denom
                };
                let s = if denom == 0.0 {
                    f64::NAN
                } else {
                    se_lp / denom
                };
                (Some(b), Some(s))
            } else {
                (None, None)
            }
        } else if let (Some(eff), Some(vv)) = (eff_i, var_i) {
            let denom = (eff * eff * vv + (std::f64::consts::PI.powi(2) / 3.0)).sqrt();
            let b = if denom == 0.0 { f64::NAN } else { eff / denom };
            let s = if se_logit {
                se_i.map(|s| if denom == 0.0 { f64::NAN } else { s / denom })
            } else {
                se_i.map(|s| {
                    if denom == 0.0 {
                        f64::NAN
                    } else {
                        (s / eff.exp()) / denom
                    }
                })
            };
            (Some(b), s)
        } else {
            (None, None)
        };

        beta_std.push(beta_val.unwrap_or(f64::NAN));
        se_std.push(se_val.unwrap_or(f64::NAN));
    }

    if !ols && !linprob && se.is_none() {
        log_line(
            log,
            &format!(
                "SE column missing in {} for logistic transformation",
                file.display()
            ),
            true,
        )?;
    }

    let mut beta_series = Series::new("BETA_STD".into(), beta_std);
    let mut se_series = Series::new("SE_STD".into(), se_std);
    beta_series.rename("BETA_STD".into());
    se_series.rename("SE_STD".into());
    df.with_column(beta_series.into())?;
    df.with_column(se_series.into())?;
    Ok(df)
}

fn average_z(df: &DataFrame) -> Result<f64> {
    let beta = df
        .column("BETA_STD")?
        .as_series()
        .context("BETA_STD")?
        .f64()?;
    let se = df.column("SE_STD")?.as_series().context("SE_STD")?.f64()?;
    let mut sum = 0.0;
    let mut count = 0.0;
    for (b, s) in beta.into_iter().zip(se) {
        if let (Some(b), Some(s)) = (b, s)
            && s != 0.0
        {
            sum += (b / s).abs();
            count += 1.0;
        }
    }
    Ok(if count > 0.0 { sum / count } else { 0.0 })
}

fn write_sumstats_output(df: &DataFrame, config: &SumstatsConfig) -> Result<()> {
    let out_name = if let Some(name) = &config.log_name {
        format!("{name}_sumstats.tsv")
    } else if let Some(names) = &config.trait_names {
        let mut joined = names.join("_");
        joined = joined.replace('/', "");
        if joined.len() > 200 {
            joined.truncate(100);
        }
        if joined.is_empty() {
            "sumstats.tsv".to_string()
        } else {
            format!("{joined}_sumstats.tsv")
        }
    } else {
        "sumstats.tsv".to_string()
    };

    let mut file = File::create(&out_name)?;
    let mut csv = CsvWriter::new(&mut file).with_separator(b'\t');
    let mut df = df.clone();
    csv.finish(&mut df)?;
    Ok(())
}

fn validate_lengths(config: &SumstatsConfig) -> Result<()> {
    let n = config.files.len();
    if n == 0 {
        return Err(anyhow::anyhow!("No input files provided"));
    }
    if config.se_logit.len() != n {
        return Err(anyhow::anyhow!(
            "se_logit length {} does not match number of files {}",
            config.se_logit.len(),
            n
        ));
    }
    if let Some(names) = &config.trait_names
        && names.len() != n
    {
        return Err(anyhow::anyhow!(
            "trait_names length {} does not match number of files {}",
            names.len(),
            n
        ));
    }
    if let Some(v) = &config.ols
        && v.len() != n
    {
        return Err(anyhow::anyhow!(
            "ols length {} does not match number of files {}",
            v.len(),
            n
        ));
    }
    if let Some(v) = &config.linprob
        && v.len() != n
    {
        return Err(anyhow::anyhow!(
            "linprob length {} does not match number of files {}",
            v.len(),
            n
        ));
    }
    if let Some(v) = &config.n
        && v.len() != n
    {
        return Err(anyhow::anyhow!(
            "N length {} does not match number of files {}",
            v.len(),
            n
        ));
    }
    if let Some(v) = &config.betas
        && v.len() != n
    {
        return Err(anyhow::anyhow!(
            "betas length {} does not match number of files {}",
            v.len(),
            n
        ));
    }
    Ok(())
}
