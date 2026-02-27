use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::GzEncoder;
use polars::prelude::*;
use tracing::{info, warn};

use crate::df_utils::{
    add_z_score, ensure_f64, ensure_utf8, filter_allele_mismatch, filter_missing, filter_non_acgt,
    flip_effect_if_needed, uppercase_alleles,
};
use crate::io::read_table;
use crate::logging::log_line;
use crate::qc::{check_equal_length, check_file_exists, check_range_f64};
use crate::schema::{ColumnMapConfig, resolve_column_map};

#[derive(Debug, Clone)]
pub struct MungeConfig {
    pub files: Vec<PathBuf>,
    pub hm3: PathBuf,
    pub trait_names: Option<Vec<String>>,
    pub n: Option<Vec<f64>>,
    pub info_filter: f64,
    pub maf_filter: f64,
    pub column_names: HashMap<String, String>,
    pub parallel: bool,
    pub cores: Option<usize>,
    pub overwrite: bool,
    pub log_name: Option<String>,
}

pub fn munge(config: &MungeConfig) -> Result<()> {
    check_file_exists(&config.hm3, "hm3")?;
    check_range_f64(config.info_filter, 0.0, 1.0, false, "info.filter")?;
    check_range_f64(config.maf_filter, 0.0, 0.5, false, "maf.filter")?;

    if let Some(names) = &config.trait_names {
        check_equal_length(config.files.len(), names.len(), "files", "trait.names")?;
    }
    if let Some(n) = &config.n {
        check_equal_length(config.files.len(), n.len(), "files", "N")?;
    }

    let mut ref_df = read_table(&config.hm3)?;
    ref_df = rename_ref_columns(ref_df)?;

    let mut log = open_log_file(config)?;
    log_line(
        &mut log,
        &format!(
            "The munging of {} summary statistics started.",
            config.files.len()
        ),
        true,
    )?;

    for (idx, file) in config.files.iter().enumerate() {
        check_file_exists(file, "files")?;
        let trait_name = resolve_trait_name(config, idx, file)?;
        log_line(&mut log, &format!("Munging file: {}", file.display()), true)?;

        let mut df = read_table(file)?;
        df = handle_neff_columns(df)?;

        let headers: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let n_val = config
            .n
            .as_ref()
            .and_then(|v| v.get(idx))
            .copied()
            .filter(|v| v.is_finite());

        let map = resolve_column_map(
            &headers,
            &ColumnMapConfig {
                userprovided: config.column_names.clone(),
                check_single: vec![
                    "P".into(),
                    "A1".into(),
                    "A2".into(),
                    "EFFECT".into(),
                    "SNP".into(),
                ],
                warn_for_missing: vec![
                    "P".into(),
                    "A1".into(),
                    "A2".into(),
                    "EFFECT".into(),
                    "SNP".into(),
                    "N".into(),
                ],
                stop_on_missing: Vec::new(),
                warn_z_as_effect: false,
                n_provided: n_val.is_some(),
                filename: Some(file.display().to_string()),
            },
        )?;

        for msg in &map.info {
            log_line(&mut log, msg, true)?;
        }
        for msg in &map.warnings {
            log_line(&mut log, msg, true)?;
        }

        df.set_column_names(&map.headers)?;

        if let Some(n_const) = n_val {
            let n_series = Series::new("N".into(), vec![n_const; df.height()]);
            df.with_column(n_series.into())?;
            log_line(
                &mut log,
                &format!("Using provided N ({}) for file {}", n_const, file.display()),
                true,
            )?;
        }

        df = ensure_utf8(df, &["SNP", "A1", "A2"])?;
        df = ensure_f64(df, &["P", "EFFECT", "INFO", "MAF", "N"])?;
        df = normalize_maf(df)?;
        df = uppercase_alleles(df)?;
        let (df_tmp, removed) = filter_non_acgt(df, "A1", "A2")?;
        df = df_tmp;
        if removed > 0 {
            log_line(
                &mut log,
                &format!(
                    "{removed} row(s) were removed from {} due to non-ACGT alleles",
                    file.display()
                ),
                true,
            )?;
        }

        let before = df.height();
        let mut df = df.join(&ref_df, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
        let after = df.height();
        log_line(
            &mut log,
            &format!(
                "{} rows removed after merging with reference",
                before.saturating_sub(after)
            ),
            true,
        )?;

        let (df_tmp, removed) = filter_missing(df, "P")?;
        df = df_tmp;
        if removed > 0 {
            log_line(
                &mut log,
                &format!(
                    "{removed} rows were removed from {} due to missing values in the P column",
                    file.display()
                ),
                true,
            )?;
        }
        let (df_tmp, removed) = filter_missing(df, "EFFECT")?;
        df = df_tmp;
        if removed > 0 {
            log_line(
                &mut log,
                &format!(
                    "{removed} rows were removed from {} due to missing values in the EFFECT column",
                    file.display()
                ),
                true,
            )?;
        }

        p_value_sanity_check(&df, &mut log, file)?;

        df = detect_or_log_transform(df)?;

        df = flip_effect_if_needed(df, "A1", "A1_REF", "A2_REF", "EFFECT")?;
        let (df_tmp, removed) = filter_allele_mismatch(df, "A1", "A2", "A1_REF", "A2_REF")?;
        df = df_tmp;
        if removed > 0 {
            log_line(
                &mut log,
                &format!(
                    "{removed} row(s) were removed from {} due to allele mismatch with reference",
                    file.display()
                ),
                true,
            )?;
        }

        df = filter_info(df, config.info_filter, &mut log)?;
        df = filter_maf(df, config.maf_filter, &mut log)?;

        df = add_z_score(df, "EFFECT", "P")?;

        if df.column("N").is_err() {
            log_line(
                &mut log,
                &format!(
                    "Cannot find sample size column and N was not provided for {}",
                    file.display()
                ),
                true,
            )?;
            let n_series = Series::new("N".into(), vec![f64::NAN; df.height()]);
            df.with_column(n_series.into())?;
        }

        let output = df.select(["SNP", "N", "Z", "A1", "A2"])?;
        write_sumstats(&output, &trait_name, config.overwrite)?;
    }

    Ok(())
}

fn resolve_trait_name(config: &MungeConfig, idx: usize, file: &Path) -> Result<String> {
    if let Some(names) = &config.trait_names {
        return Ok(names[idx].clone());
    }
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("trait")
        .to_string();
    Ok(stem)
}

fn open_log_file(config: &MungeConfig) -> Result<File> {
    let log_name = if let Some(name) = &config.log_name {
        name.clone()
    } else if let Some(names) = &config.trait_names {
        let mut joined = names.join("_");
        joined = joined.replace('/', "");
        if joined.len() > 200 {
            joined.truncate(100);
        }
        if joined.is_empty() {
            "munge".to_string()
        } else {
            joined
        }
    } else {
        "munge".to_string()
    };
    let path = format!("{log_name}_munge.log");
    let file = File::create(path)?;
    Ok(file)
}

fn rename_ref_columns(mut df: DataFrame) -> Result<DataFrame> {
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

fn handle_neff_columns(mut df: DataFrame) -> Result<DataFrame> {
    if df.column("NEFFDIV2").is_ok() {
        let neff = df
            .column("NEFFDIV2")?
            .as_series()
            .context("NEFFDIV2")?
            .f64()?
            .clone()
            * 2.0;
        let mut series = neff.into_series();
        series.rename("NEFF".into());
        df.with_column(series.into())?;
        df.drop_in_place("NEFFDIV2")?;
    }
    if df.column("NEFF_HALF").is_ok() {
        let neff = df
            .column("NEFF_HALF")?
            .as_series()
            .context("NEFF_HALF")?
            .f64()?
            .clone()
            * 2.0;
        let mut series = neff.into_series();
        series.rename("NEFF".into());
        df.with_column(series.into())?;
        df.drop_in_place("NEFF_HALF")?;
    }
    Ok(df)
}

fn normalize_maf(mut df: DataFrame) -> Result<DataFrame> {
    if df.column("MAF").is_ok() {
        let maf = df
            .column("MAF")?
            .as_series()
            .context("MAF")?
            .f64()?
            .apply(|v| v.map(|x| if x > 0.5 { 1.0 - x } else { x }))
            .into_series();
        let mut series = maf;
        series.rename("MAF".into());
        df.with_column(series.into())?;
    }
    Ok(df)
}

fn detect_or_log_transform(mut df: DataFrame) -> Result<DataFrame> {
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
        info!("Effect column appears to be OR; applied log transform.");
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
            &format!("{removed} rows were removed due to INFO values below {info_filter}"),
            true,
        )?;
    }
    Ok(df)
}

fn filter_maf(mut df: DataFrame, maf_filter: f64, log: &mut File) -> Result<DataFrame> {
    if df.column("MAF").is_err() {
        log_line(log, "No MAF column, cannot filter on MAF", true)?;
        return Ok(df);
    }
    let before = df.height();
    let maf = df.column("MAF")?.as_series().context("MAF")?.f64()?;
    let mask = maf.gt_eq(maf_filter);
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    if removed > 0 {
        log_line(
            log,
            &format!("{removed} rows were removed due to MAF below {maf_filter}"),
            true,
        )?;
    }
    Ok(df)
}

fn p_value_sanity_check(df: &DataFrame, log: &mut File, file: &Path) -> Result<()> {
    if df.column("P").is_err() {
        return Ok(());
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
                "In excess of 100 SNPs have P val above 1 or below 0 in {}. The P column may be mislabeled.",
                file.display()
            ),
            true,
        )?;
    }
    Ok(())
}

fn write_sumstats(df: &DataFrame, trait_name: &str, overwrite: bool) -> Result<()> {
    let out_name = format!("{trait_name}.sumstats.gz");
    if !overwrite && Path::new(&out_name).exists() {
        warn!("{} exists and overwrite=false; skipping", out_name);
        return Ok(());
    }

    let file = File::create(&out_name)?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut writer = std::io::BufWriter::new(encoder);

    let mut csv = CsvWriter::new(&mut writer).with_separator(b'\t');
    let mut df = df.clone();
    csv.finish(&mut df)?;

    writer.flush()?;
    Ok(())
}
