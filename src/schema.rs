use std::collections::{HashMap, HashSet};

use crate::error::{GenomicSemError, Result};

#[derive(Debug, Clone)]
pub struct ColumnMap {
    pub headers: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ColumnMapConfig {
    pub userprovided: HashMap<String, String>,
    pub check_single: Vec<String>,
    pub warn_for_missing: Vec<String>,
    pub stop_on_missing: Vec<String>,
    pub warn_z_as_effect: bool,
    pub n_provided: bool,
    pub filename: Option<String>,
}

pub fn normalize_headers(headers: &[String]) -> Vec<String> {
    headers
        .iter()
        .map(|h| h.trim().to_ascii_uppercase())
        .collect()
}

pub fn resolve_column_map(headers: &[String], config: &ColumnMapConfig) -> Result<ColumnMap> {
    let mut warnings = Vec::new();
    let mut info = Vec::new();

    let mut headers = normalize_headers(headers);

    let filename = config
        .filename
        .clone()
        .unwrap_or_else(|| "<unknown>".to_string());

    let mut synonyms: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    synonyms.insert(
        "SNP",
        vec![
            "SNP",
            "SNPID",
            "RSID",
            "RS_NUMBER",
            "RS_NUMBERS",
            "MARKERNAME",
            "ID",
            "PREDICTOR",
            "SNP_ID",
            "VARIANTID",
            "VARIANT_ID",
            "RSIDS",
            "RS_ID",
        ],
    );
    synonyms.insert(
        "A1",
        vec![
            "A1",
            "ALLELE1",
            "EFFECT_ALLELE",
            "INC_ALLELE",
            "REFERENCE_ALLELE",
            "EA",
            "REF",
        ],
    );
    synonyms.insert(
        "A2",
        vec![
            "A2",
            "ALLELE2",
            "ALLELE0",
            "OTHER_ALLELE",
            "NON_EFFECT_ALLELE",
            "DEC_ALLELE",
            "OA",
            "NEA",
            "ALT",
            "A0",
        ],
    );
    synonyms.insert(
        "EFFECT",
        vec![
            "OR",
            "B",
            "BETA",
            "LOG_ODDS",
            "EFFECTS",
            "EFFECT",
            "SIGNED_SUMSTAT",
            "EST",
            "BETA1",
            "LOGOR",
        ],
    );
    synonyms.insert("INFO", vec!["INFO", "IMPINFO"]);
    synonyms.insert(
        "P",
        vec![
            "P",
            "PVALUE",
            "PVAL",
            "P_VALUE",
            "P-VALUE",
            "P.VALUE",
            "P_VAL",
            "GC_PVALUE",
            "WALD_P",
        ],
    );
    synonyms.insert(
        "N",
        vec![
            "N",
            "WEIGHT",
            "NCOMPLETESAMPLES",
            "TOTALSAMPLESIZE",
            "TOTALN",
            "TOTAL_N",
            "N_COMPLETE_SAMPLES",
            "SAMPLESIZE",
            "NEFF",
            "N_EFF",
            "N_EFFECTIVE",
            "SUMNEFF",
        ],
    );
    synonyms.insert(
        "MAF",
        vec![
            "MAF",
            "CEUAF",
            "FREQ1",
            "EAF",
            "FREQ1.HAPMAP",
            "FREQALLELE1HAPMAPCEU",
            "FREQ.ALLELE1.HAPMAPCEU",
            "EFFECT_ALLELE_FREQ",
            "FREQ.A1",
            "A1FREQ",
            "ALLELEFREQ",
            "EFFECT_ALLELE_FREQUENCY",
        ],
    );
    synonyms.insert(
        "Z",
        vec![
            "Z",
            "ZSCORE",
            "Z-SCORE",
            "ZSTATISTIC",
            "ZSTAT",
            "Z-STATISTIC",
        ],
    );
    synonyms.insert(
        "SE",
        vec![
            "STDERR",
            "SE",
            "STDERRLOGOR",
            "SEBETA",
            "STANDARDERROR",
            "STANDARD_ERROR",
        ],
    );
    synonyms.insert("DIRECTION", vec!["DIRECTION", "DIREC", "DIRE", "SIGN"]);

    let full_names: HashMap<&'static str, &'static str> = [
        ("P", "P-value"),
        ("A1", "effect allele"),
        ("A2", "other allele"),
        ("EFFECT", "beta or effect"),
        ("SNP", "rs-id"),
        ("SE", "standard error"),
        ("DIRECTION", "direction"),
    ]
    .into_iter()
    .collect();

    if headers.iter().any(|h| h == "ALT") && headers.iter().any(|h| h == "REF") {
        info.push(format!(
            "Found REF and ALT columns in the summary statistic file {filename}. REF will be interpreted as A1 and ALT as A2."
        ));
    }

    if !config.n_provided {
        let neff_cols = ["NEFF", "N_EFF", "N_EFFECTIVE", "SUMNEFF"];
        if headers.iter().any(|h| neff_cols.contains(&h.as_str())) {
            info.push("Found an NEFF column for sample size. This is likely effective sample size and should only be used for liability h^2 conversion for binary traits. If Neff is halved and not recognized, it should be manually doubled before munging.".to_string());
        }
    }

    let mut stop_on_missing = HashSet::new();
    for col in &config.stop_on_missing {
        stop_on_missing.insert(col.to_ascii_uppercase());
    }

    let mut warn_for_missing = HashSet::new();
    for col in &config.warn_for_missing {
        warn_for_missing.insert(col.to_ascii_uppercase());
    }

    let mut user_map: HashMap<String, String> = HashMap::new();
    for (k, v) in &config.userprovided {
        user_map.insert(k.to_ascii_uppercase(), v.to_ascii_uppercase());
    }

    for (canonical, syns) in synonyms.iter() {
        if *canonical == "N" && config.n_provided {
            continue;
        }

        if let Some(user_col) = user_map.get(*canonical) {
            let mut matched = false;
            for h in headers.iter_mut() {
                if h == user_col {
                    *h = canonical.to_string();
                    matched = true;
                }
            }
            if matched {
                info.push(format!(
                    "Interpreting the {user_col} column as the {canonical} column, as requested."
                ));
            }
            continue;
        }

        if headers.iter().any(|h| h == *canonical) {
            info.push(format!(
                "Interpreting the {canonical} column as the {canonical} column."
            ));
            continue;
        }

        let mut matched = false;
        for h in headers.iter_mut() {
            if syns.contains(&h.as_str()) {
                *h = canonical.to_string();
                matched = true;
            }
        }
        if matched {
            let matched_cols: Vec<String> = syns
                .iter()
                .filter(|s| headers.iter().any(|h| h == *s))
                .map(|s| s.to_string())
                .collect();
            let shown = if matched_cols.is_empty() {
                "<synonym>".to_string()
            } else {
                matched_cols.join(", ")
            };
            info.push(format!(
                "Interpreting the {shown} column as the {canonical} column."
            ));
            continue;
        }

        if *canonical == "EFFECT" {
            let z_present = headers.iter().any(|h| h == "Z");
            if z_present {
                if !config.warn_z_as_effect {
                    for h in headers.iter_mut() {
                        if h == "Z" {
                            *h = "EFFECT".to_string();
                        }
                    }
                    info.push("Interpreting the Z column as the EFFECT column.".to_string());
                } else {
                    warnings.push(format!(
                        "Z-statistic column detected in {filename}. Set linprob/OLS or remove Z if betas are available."
                    ));
                }
            }
            continue;
        }

        if warn_for_missing.contains(*canonical) {
            warnings.push(format!(
                "Cannot find {canonical} column; try renaming it to {canonical} in {filename}."
            ));
        } else if stop_on_missing.contains(*canonical) {
            return Err(GenomicSemError::MissingColumn(canonical.to_string()));
        }
    }

    for col in &config.check_single {
        let col = col.to_ascii_uppercase();
        let count = headers.iter().filter(|h| h.as_str() == col).count();
        if count == 0 {
            if let Some(full) = full_names.get(col.as_str()) {
                warnings.push(format!(
                    "Cannot find {full} column; try renaming it to {col} in {filename}."
                ));
            } else {
                warnings.push(format!(
                    "Cannot find {col} column; try renaming it to {col} in {filename}."
                ));
            }
        }
        if count > 1 {
            if let Some(full) = full_names.get(col.as_str()) {
                warnings.push(format!(
                    "Multiple columns interpreted as {full}; rename the one you don't want interpreted as {col}2 in {filename}."
                ));
            } else {
                warnings.push(format!(
                    "Multiple columns interpreted as {col}; rename the one you don't want interpreted as {col}2 in {filename}."
                ));
            }
        }
    }

    Ok(ColumnMap {
        headers,
        warnings,
        info,
    })
}
