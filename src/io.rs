use std::borrow::Cow;
use std::fs::{File, read_to_string};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use polars::prelude::*;
use tempfile::NamedTempFile;

use crate::types::{LdscOutput, Matrix, StratifiedLdscOutput};

pub fn read_table(path: &Path) -> Result<DataFrame> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "gz" || ext == "bz2" {
        let tmp = decompress_to_temp(path, &ext)?;
        return read_table_plain(tmp.path());
    }

    read_table_plain(path)
}

fn read_table_plain(path: &Path) -> Result<DataFrame> {
    let delimiter = detect_delimiter(path)?;
    if delimiter == b' ' {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        return read_table_whitespace(BufReader::new(file));
    }

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(
            CsvParseOptions::default()
                .with_separator(delimiter)
                .with_null_values(Some(NullValues::AllColumns(vec![
                    "".into(),
                    "NA".into(),
                    "NaN".into(),
                    ".".into(),
                ])))
                .with_missing_is_null(true),
        )
        .with_ignore_errors(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()
        .with_context(|| format!("read {}", path.display()))?;
    trim_string_columns(df)
}

fn read_table_whitespace<R: Read>(reader: R) -> Result<DataFrame> {
    let mut reader = BufReader::new(reader);
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;
    if header_line.trim().is_empty() {
        return Err(anyhow::anyhow!("empty file"));
    }
    let headers = split_quoted_whitespace(&header_line);
    let mut columns: Vec<Vec<String>> = vec![Vec::new(); headers.len()];

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parts = split_quoted_whitespace(&line);
        for (i, col) in columns.iter_mut().enumerate() {
            let value = parts.get(i).cloned().unwrap_or_default();
            let value = normalize_missing_token(&value);
            col.push(value);
        }
    }

    let series: Vec<Series> = headers
        .iter()
        .zip(columns)
        .map(|(name, values)| Series::new(name.as_str().into(), values))
        .collect();
    let height = series.first().map(|s| s.len()).unwrap_or(0);
    let cols: Vec<Column> = series.into_iter().map(Into::into).collect();
    let df = DataFrame::new(height, cols)?;
    trim_string_columns(df)
}

fn detect_delimiter(path: &Path) -> Result<u8> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut first = String::new();
    reader.read_line(&mut first)?;
    if first.contains('\t') {
        return Ok(b'\t');
    }
    if first.contains(',') {
        return Ok(b',');
    }
    Ok(b' ')
}

fn decompress_to_temp(path: &Path, ext: &str) -> Result<NamedTempFile> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut decoder: Box<dyn Read> = match ext {
        "gz" => Box::new(GzDecoder::new(file)),
        "bz2" => Box::new(BzDecoder::new(file)),
        _ => Box::new(file),
    };
    let mut tmp = NamedTempFile::new()?;
    std::io::copy(&mut decoder, &mut tmp)?;
    Ok(tmp)
}

pub fn uppercase_series(series: &Series) -> Result<Series> {
    let utf8 = series.str()?;
    let upper = utf8
        .apply(|v| v.map(|s| Cow::Owned(s.to_ascii_uppercase())))
        .into_series();
    Ok(upper)
}

fn trim_series(series: &Series) -> Result<Series> {
    let utf8 = series.str()?;
    let trimmed = utf8
        .apply(|v| v.map(|s| Cow::Owned(s.trim().to_string())))
        .into_series();
    Ok(trimmed)
}

fn trim_string_columns(mut df: DataFrame) -> Result<DataFrame> {
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    for name in names {
        if let Ok(column) = df.column(&name)
            && column.dtype() == &DataType::String
        {
            let trimmed = trim_series(column.as_series().context("series")?)?;
            let mut s = trimmed;
            s.rename(name.clone().into());
            df.with_column(s.into())?;
        }
    }
    Ok(df)
}

fn split_quoted_whitespace(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            in_quote = !in_quote;
            continue;
        }
        if c == '\\'
            && let Some('"') = chars.peek().copied()
        {
            chars.next();
            current.push('"');
            continue;
        }
        if c.is_whitespace() && !in_quote {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn normalize_missing_token(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let upper = trimmed.to_ascii_uppercase();
    if upper == "NA" || upper == "NAN" || trimmed == "." {
        String::new()
    } else {
        trimmed.to_string()
    }
}

pub fn read_matrix_file(path: &Path) -> Result<Matrix> {
    let df = read_table(path)?;
    let n = df.height();
    let m = df.width();
    let mut out = vec![vec![f64::NAN; m]; n];
    let names = df.get_column_names();
    for (j, name) in names.iter().enumerate() {
        let series = df
            .column(name)?
            .as_series()
            .context("series")?
            .cast(&DataType::Float64)?;
        let col = series.f64()?;
        for (i, row) in out.iter_mut().enumerate().take(n) {
            row[j] = col.get(i).unwrap_or(f64::NAN);
        }
    }
    Ok(out)
}

pub fn read_vector_file(path: &Path) -> Result<Vec<f64>> {
    let df = read_table(path)?;
    let name = df
        .get_column_names()
        .first()
        .ok_or_else(|| anyhow::anyhow!("Empty vector file {}", path.display()))?
        .to_string();
    let series = df
        .column(&name)?
        .as_series()
        .context("series")?
        .cast(&DataType::Float64)?;
    let col = series.f64()?;
    Ok(col.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

pub fn write_matrix<T: std::fmt::Display>(matrix: &[Vec<T>], path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    for row in matrix {
        let line = row
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\t");
        writeln!(file, "{line}")?;
    }
    Ok(())
}

pub fn write_vector<T: std::fmt::Display>(vec: &[T], path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    for v in vec {
        writeln!(file, "{v}")?;
    }
    Ok(())
}

pub fn write_scalar<T: std::fmt::Display>(value: T, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{value}")?;
    Ok(())
}

pub fn write_dataframe(df: &DataFrame, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    let mut csv = CsvWriter::new(&mut file).with_separator(b'\t');
    let mut df = df.clone();
    csv.finish(&mut df)?;
    Ok(())
}

pub fn write_ldsc_json(out: &LdscOutput, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    let json = format!(
        "{{\"trait_names\":{},\"S\":{},\"V\":{},\"I\":{},\"N\":{},\"m\":{},\"S_stand\":{},\"V_stand\":{}}}",
        format_string_vec(&out.trait_names),
        format_matrix(&out.s),
        format_matrix(&out.v),
        format_matrix(&out.i),
        format_vec(&out.n),
        out.m,
        format_option_matrix(&out.s_stand),
        format_option_matrix(&out.v_stand),
    );
    writeln!(file, "{json}")?;
    Ok(())
}

pub fn read_ldsc_json(path: &Path) -> Result<LdscOutput> {
    let text = read_to_string(path).context("read ldsc json")?;
    let trait_names = parse_string_array(&extract_json_value(&text, "trait_names")?)?;
    let s = parse_matrix(&extract_json_value(&text, "S")?)?;
    let v = parse_matrix(&extract_json_value(&text, "V")?)?;
    let i = parse_matrix(&extract_json_value(&text, "I")?)?;
    let n = parse_number_array(&extract_json_value(&text, "N")?)?;
    let m = extract_json_value(&text, "m")?
        .parse::<f64>()
        .context("parse m")? as usize;
    let s_stand = parse_optional_matrix(&extract_json_value(&text, "S_stand")?)?;
    let v_stand = parse_optional_matrix(&extract_json_value(&text, "V_stand")?)?;
    Ok(LdscOutput {
        s,
        v,
        i,
        n,
        m,
        s_stand,
        v_stand,
        trait_names,
    })
}

pub fn write_s_ldsc_json(out: &StratifiedLdscOutput, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    let json = format!(
        "{{\"trait_names\":{},\"annotation_names\":{},\"S\":{},\"V\":{},\"S_Tau\":{},\"V_Tau\":{},\"I\":{},\"N\":{},\"m\":{},\"Prop\":{},\"Select\":{}}}",
        format_string_vec(&out.trait_names),
        format_string_vec(&out.annotation_names),
        format_matrix_list(&out.s),
        format_matrix_list(&out.v),
        format_matrix_list(&out.s_tau),
        format_matrix_list(&out.v_tau),
        format_matrix(&out.i),
        format_vec(&out.n),
        format_vec(&out.m),
        format_vec(&out.prop),
        format_i32_vec(&out.select),
    );
    writeln!(file, "{json}")?;
    Ok(())
}

pub fn read_s_ldsc_json(path: &Path) -> Result<StratifiedLdscOutput> {
    let text = read_to_string(path).context("read s_ldsc json")?;
    let trait_names = parse_string_array(&extract_json_value(&text, "trait_names")?)?;
    let annotation_names = parse_string_array(&extract_json_value(&text, "annotation_names")?)?;
    let s = parse_matrix_list(&extract_json_value(&text, "S")?)?;
    let v = parse_matrix_list(&extract_json_value(&text, "V")?)?;
    let s_tau = parse_matrix_list(&extract_json_value(&text, "S_Tau")?)?;
    let v_tau = parse_matrix_list(&extract_json_value(&text, "V_Tau")?)?;
    let i = parse_matrix(&extract_json_value(&text, "I")?)?;
    let n = parse_number_array(&extract_json_value(&text, "N")?)?;
    let m = parse_number_array(&extract_json_value(&text, "m")?)?;
    let prop = parse_number_array(&extract_json_value(&text, "Prop")?)?;
    let select_raw = parse_number_array(&extract_json_value(&text, "Select")?)?;
    let select = select_raw
        .into_iter()
        .map(|v| if v.is_finite() { v as i32 } else { 0 })
        .collect();
    Ok(StratifiedLdscOutput {
        s,
        v,
        s_tau,
        v_tau,
        i,
        n,
        m,
        prop,
        select,
        annotation_names,
        trait_names,
    })
}

fn extract_json_value(text: &str, key: &str) -> Result<String> {
    let needle = format!("\"{key}\":");
    let start = text
        .find(&needle)
        .ok_or_else(|| anyhow::anyhow!("Key {key} not found in JSON"))?;
    let mut idx = start + needle.len();
    let bytes = text.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() {
        return Err(anyhow::anyhow!("Key {key} has no value"));
    }
    if bytes[idx] == b'[' {
        let mut depth = 0isize;
        for j in idx..bytes.len() {
            match bytes[j] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(text[idx..=j].to_string());
                    }
                }
                _ => {}
            }
        }
        Err(anyhow::anyhow!("Unterminated array for key {key}"))
    } else {
        let mut j = idx;
        while j < bytes.len() && bytes[j] != b',' && bytes[j] != b'}' {
            j += 1;
        }
        Ok(text[idx..j].trim().to_string())
    }
}

fn parse_optional_matrix(value: &str) -> Result<Option<Matrix>> {
    let trimmed = value.trim();
    if trimmed == "null" {
        return Ok(None);
    }
    Ok(Some(parse_matrix(trimmed)?))
}

fn parse_matrix(value: &str) -> Result<Matrix> {
    let arrays = split_top_level_arrays(value)?;
    let mut out = Vec::with_capacity(arrays.len());
    for arr in arrays {
        out.push(parse_number_array(arr)?);
    }
    Ok(out)
}

fn parse_matrix_list(value: &str) -> Result<Vec<Matrix>> {
    let arrays = split_top_level_arrays(value)?;
    let mut out = Vec::with_capacity(arrays.len());
    for arr in arrays {
        out.push(parse_matrix(arr)?);
    }
    Ok(out)
}

fn split_top_level_arrays(value: &str) -> Result<Vec<&str>> {
    let trimmed = value.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(anyhow::anyhow!("Expected array"));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = Vec::new();
    let mut depth = 0isize;
    let mut start = None;
    for (idx, ch) in inner.char_indices() {
        if ch == '[' {
            if depth == 0 {
                start = Some(idx);
            }
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start {
                    out.push(inner[st..=idx].trim());
                }
                start = None;
            }
        }
    }
    if depth != 0 {
        return Err(anyhow::anyhow!("Unbalanced brackets in array"));
    }
    Ok(out)
}

fn parse_number_array(value: &str) -> Result<Vec<f64>> {
    let trimmed = value.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(anyhow::anyhow!("Expected numeric array"));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let token = part.trim();
        if token == "null" {
            out.push(f64::NAN);
        } else {
            out.push(token.parse::<f64>().context("parse number")?);
        }
    }
    Ok(out)
}

fn parse_string_array(value: &str) -> Result<Vec<String>> {
    let trimmed = value.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(anyhow::anyhow!("Expected string array"));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = Vec::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.peek().copied() {
        if ch.is_whitespace() || ch == ',' {
            chars.next();
            continue;
        }
        if ch == '"' {
            chars.next();
            let mut buf = String::new();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    if let Some(esc) = chars.next() {
                        buf.push(match esc {
                            '\\' => '\\',
                            '"' => '"',
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                    }
                } else if c == '"' {
                    break;
                } else {
                    buf.push(c);
                }
            }
            out.push(buf);
        } else {
            return Err(anyhow::anyhow!("Unexpected token in string array"));
        }
    }
    Ok(out)
}

fn format_option_matrix(matrix: &Option<Vec<Vec<f64>>>) -> String {
    match matrix {
        Some(m) => format_matrix(m),
        None => "null".to_string(),
    }
}

fn format_matrix(matrix: &[Vec<f64>]) -> String {
    let rows = matrix
        .iter()
        .map(|row| format_vec(row))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rows}]")
}

fn format_matrix_list(matrices: &[Vec<Vec<f64>>]) -> String {
    let entries = matrices
        .iter()
        .map(|m| format_matrix(m))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{entries}]")
}

fn format_vec(values: &[f64]) -> String {
    let vals = values
        .iter()
        .map(|v| format_number(*v))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{vals}]")
}

fn format_i32_vec(values: &[i32]) -> String {
    let vals = values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{vals}]")
}

fn format_string_vec(values: &[String]) -> String {
    let vals = values
        .iter()
        .map(|v| format!("\"{}\"", escape_json_string(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{vals}]")
}

fn escape_json_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn format_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.15}")
    } else {
        "null".to_string()
    }
}
