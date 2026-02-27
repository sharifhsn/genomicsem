use anyhow::{Context, Result};
use polars::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::borrow::Cow;

pub fn ensure_utf8(mut df: DataFrame, cols: &[&str]) -> Result<DataFrame> {
    for col in cols {
        if let Ok(column) = df.column(col)
            && let Some(series) = column.as_series()
            && series.dtype() != &DataType::String
        {
            let mut casted = series.cast(&DataType::String)?;
            casted.rename((*col).into());
            df.with_column(casted.into())?;
        }
    }
    Ok(df)
}

pub fn ensure_f64(mut df: DataFrame, cols: &[&str]) -> Result<DataFrame> {
    for col in cols {
        if let Ok(column) = df.column(col)
            && let Some(series) = column.as_series()
            && series.dtype() != &DataType::Float64
        {
            let mut casted = series.cast(&DataType::Float64)?;
            casted.rename((*col).into());
            df.with_column(casted.into())?;
        }
    }
    Ok(df)
}

pub fn uppercase_alleles(mut df: DataFrame) -> Result<DataFrame> {
    for col in ["A1", "A2"] {
        if let Ok(column) = df.column(col)
            && let Some(series) = column.as_series()
            && let Ok(utf8) = series.str()
        {
            let upper = utf8
                .apply(|v| v.map(|s| Cow::Owned(s.to_ascii_uppercase())))
                .into_series();
            let mut s = upper;
            s.rename(col.into());
            df.with_column(s.into())?;
        }
    }
    Ok(df)
}

pub fn filter_non_acgt(
    mut df: DataFrame,
    a1_col: &str,
    a2_col: &str,
) -> Result<(DataFrame, usize)> {
    if df.column(a1_col).is_err() || df.column(a2_col).is_err() {
        return Ok((df, 0));
    }
    let before = df.height();
    let a1 = df.column(a1_col)?.as_series().context("A1")?.str()?;
    let a2 = df.column(a2_col)?.as_series().context("A2")?.str()?;
    let mask: BooleanChunked = a1
        .into_iter()
        .zip(a2)
        .map(|(a1v, a2v)| match (a1v, a2v) {
            (Some(x), Some(y)) => {
                let ok1 = matches!(x, "A" | "C" | "G" | "T");
                let ok2 = matches!(y, "A" | "C" | "G" | "T");
                ok1 && ok2
            }
            _ => false,
        })
        .collect();
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    Ok((df, removed))
}

pub fn filter_allele_mismatch(
    mut df: DataFrame,
    a1_col: &str,
    a2_col: &str,
    a1_ref: &str,
    a2_ref: &str,
) -> Result<(DataFrame, usize)> {
    if df.column(a1_col).is_err()
        || df.column(a2_col).is_err()
        || df.column(a1_ref).is_err()
        || df.column(a2_ref).is_err()
    {
        return Ok((df, 0));
    }

    let a1 = df.column(a1_col)?.as_series().context("A1")?.str()?;
    let a2 = df.column(a2_col)?.as_series().context("A2")?.str()?;
    let a1r = df.column(a1_ref)?.as_series().context("A1_REF")?.str()?;
    let a2r = df.column(a2_ref)?.as_series().context("A2_REF")?.str()?;

    let mask: BooleanChunked = a1
        .into_iter()
        .zip(a2)
        .zip(a1r)
        .zip(a2r)
        .map(|(((a1v, a2v), a1rv), a2rv)| match (a1v, a2v, a1rv, a2rv) {
            (Some(a1v), Some(a2v), Some(a1rv), Some(a2rv)) => {
                let a1_ok = a1v == a1rv || a1v == a2rv;
                let a2_ok = a2v == a2rv || a2v == a1rv;
                a1_ok && a2_ok
            }
            _ => false,
        })
        .collect();

    let before = df.height();
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    Ok((df, removed))
}

pub fn flip_effect_if_needed(
    mut df: DataFrame,
    a1_col: &str,
    a1_ref: &str,
    a2_ref: &str,
    effect_col: &str,
) -> Result<DataFrame> {
    if df.column(a1_col).is_err()
        || df.column(a1_ref).is_err()
        || df.column(a2_ref).is_err()
        || df.column(effect_col).is_err()
    {
        return Ok(df);
    }
    let a1 = df.column(a1_col)?.as_series().context("A1")?.str()?;
    let a1_ref = df.column(a1_ref)?.as_series().context("A1_REF")?.str()?;
    let a2_ref = df.column(a2_ref)?.as_series().context("A2_REF")?.str()?;
    let effect = df
        .column(effect_col)?
        .as_series()
        .context("EFFECT")?
        .f64()?;

    let flipped = effect
        .into_iter()
        .zip(a1)
        .zip(a1_ref)
        .zip(a2_ref)
        .map(|(((eff, a1), a1r), a2r)| match (eff, a1, a1r, a2r) {
            (Some(e), Some(a1v), Some(a1rv), Some(a2rv)) => {
                if a1v != a1rv && a1v == a2rv {
                    Some(-e)
                } else {
                    Some(e)
                }
            }
            _ => None,
        })
        .collect::<Float64Chunked>()
        .into_series();

    let mut series = flipped;
    series.rename(effect_col.into());
    df.with_column(series.into())?;
    Ok(df)
}

pub fn add_z_score(mut df: DataFrame, effect_col: &str, p_col: &str) -> Result<DataFrame> {
    if df.column(effect_col).is_err() || df.column(p_col).is_err() {
        return Ok(df);
    }
    let effect = df
        .column(effect_col)?
        .as_series()
        .context("EFFECT")?
        .f64()?;
    let pvals = df.column(p_col)?.as_series().context("P")?.f64()?;
    let chi = ChiSquared::new(1.0).unwrap();

    let z: Float64Chunked = effect
        .into_iter()
        .zip(pvals)
        .map(|(eff, p)| match (eff, p) {
            (Some(e), Some(p)) if p > 0.0 && p <= 1.0 => {
                let q = chi.inverse_cdf(1.0 - p);
                Some(e.signum() * q.sqrt())
            }
            (Some(e), Some(0.0)) => Some(e.signum() * f64::INFINITY),
            _ => None,
        })
        .collect();

    let mut series = z.into_series();
    series.rename("Z".into());
    df.with_column(series.into())?;
    Ok(df)
}

pub fn filter_missing(mut df: DataFrame, col: &str) -> Result<(DataFrame, usize)> {
    if df.column(col).is_err() {
        return Ok((df, 0));
    }
    let before = df.height();
    let column = df.column(col)?;
    let series = column.as_series().context("series")?;
    let mask = match series.dtype() {
        DataType::Float64 => series.is_not_null() & series.f64()?.is_not_nan(),
        _ => series.is_not_null(),
    };
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    Ok((df, removed))
}
