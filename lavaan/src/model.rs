use std::collections::{HashMap, HashSet};

use crate::parser::{CoefSpec, Constraint, DefineLine, ModelOp, ModelSpec, parse_expr};
use crate::types::{Estimation, Matrix, ParTableRow, SemInput};
use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatKind {
    A,
    S,
}

#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub kind: MatKind,
    pub row: usize,
    pub col: usize,
    pub symmetric: bool,
    pub free_idx: usize,
    pub fixed: Option<f64>,
    pub label: Option<String>,
    pub start: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct FreeParam {
    pub label: Option<String>,
    pub start: f64,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SemModel {
    pub var_names: Vec<String>,
    pub obs_names: Vec<String>,
    pub latent_names: Vec<String>,
    pub slots: Vec<ParamSlot>,
    pub free: Vec<FreeParam>,
    pub s_obs: Matrix,
    pub gamma: Matrix,
    pub wls_v: Matrix,
    pub s_vec: Vec<f64>,
    pub w_diag: Vec<f64>,
    pub constraints: Vec<Constraint>,
    pub param_key_map: HashMap<String, usize>,
    pub label_map: HashMap<String, usize>,
    pub estimation: crate::types::Estimation,
}

impl SemModel {
    pub fn build_model(input: &SemInput, spec: &ModelSpec) -> Result<SemModel> {
        build_model(input, spec)
    }

    pub fn build_model_from_table(
        input: &SemInput,
        table: &[ParTableRow],
    ) -> Result<(SemModel, ModelSpec)> {
        build_model_from_table(input, table)
    }
}

pub fn build_model(input: &SemInput, spec: &ModelSpec) -> Result<SemModel> {
    let obs_names = input.names.clone();
    let obs_set: HashSet<String> = obs_names.iter().cloned().collect();

    let mut latent_names = Vec::new();
    for line in &spec.lines {
        if let ModelOp::Measure = line.op
            && !obs_set.contains(&line.lhs)
            && !latent_names.contains(&line.lhs)
        {
            latent_names.push(line.lhs.clone());
        }
    }

    let mut var_names = obs_names.clone();
    for latent in &latent_names {
        if !var_names.contains(latent) {
            var_names.push(latent.clone());
        }
    }

    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in var_names.iter().enumerate() {
        name_to_idx.insert(name.clone(), i);
    }

    let mut slots: Vec<ParamSlot> = Vec::new();

    let std_lv = input.std_lv;

    // Track regressions into latent variables for exogenous detection.
    let mut latent_has_incoming: HashMap<String, bool> = HashMap::new();
    for name in &latent_names {
        latent_has_incoming.insert(name.clone(), false);
    }

    for line in &spec.lines {
        match line.op {
            ModelOp::Measure => {
                let lhs = line.lhs.trim().to_string();
                let mut first = true;
                for term in &line.terms {
                    let rhs = term.var.trim().to_string();
                    if rhs == "1" {
                        continue;
                    }
                    let mut coef = term.coef.clone();
                    if first && matches!(coef, CoefSpec::None) && !std_lv {
                        coef = CoefSpec::Fixed(1.0);
                    }
                    first = false;
                    let row = *name_to_idx.get(&rhs).context("unknown observed variable")?;
                    let col = *name_to_idx.get(&lhs).context("unknown latent variable")?;
                    slots.push(make_slot(&lhs, "=~", &rhs, MatKind::A, row, col, coef));
                }
            }
            ModelOp::Regress => {
                let lhs = line.lhs.trim().to_string();
                let row = match name_to_idx.get(&lhs) {
                    Some(v) => *v,
                    None => {
                        if !latent_names.contains(&lhs) {
                            latent_names.push(lhs.clone());
                            var_names.push(lhs.clone());
                            let idx = var_names.len() - 1;
                            name_to_idx.insert(lhs.clone(), idx);
                            latent_has_incoming.insert(lhs.clone(), false);
                        }
                        *name_to_idx.get(&lhs).unwrap()
                    }
                };

                for term in &line.terms {
                    let rhs = term.var.trim().to_string();
                    if rhs == "1" {
                        continue; // intercept not supported
                    }
                    let col = match name_to_idx.get(&rhs) {
                        Some(v) => *v,
                        None => {
                            if !latent_names.contains(&rhs) && !obs_set.contains(&rhs) {
                                latent_names.push(rhs.clone());
                                var_names.push(rhs.clone());
                                let idx = var_names.len() - 1;
                                name_to_idx.insert(rhs.clone(), idx);
                                latent_has_incoming.insert(rhs.clone(), false);
                            }
                            *name_to_idx.get(&rhs).unwrap()
                        }
                    };
                    if latent_names.contains(&lhs) {
                        latent_has_incoming.insert(lhs.clone(), true);
                    }
                    slots.push(make_slot(
                        &lhs,
                        "~",
                        &rhs,
                        MatKind::A,
                        row,
                        col,
                        term.coef.clone(),
                    ));
                }
            }
            ModelOp::Cov => {
                let lhs = line.lhs.trim().to_string();
                let row = *name_to_idx.get(&lhs).context("unknown variable")?;
                for term in &line.terms {
                    let rhs = term.var.trim().to_string();
                    if rhs == "1" {
                        continue;
                    }
                    let col = *name_to_idx.get(&rhs).context("unknown variable")?;
                    let symmetric = row != col;
                    slots.push(
                        make_slot(&lhs, "~~", &rhs, MatKind::S, row, col, term.coef.clone())
                            .with_symmetric(symmetric),
                    );
                }
            }
        }
    }

    // Add default variances if missing
    for name in &var_names {
        let idx = *name_to_idx.get(name).unwrap();
        let has_var = slots
            .iter()
            .any(|s| s.kind == MatKind::S && s.row == idx && s.col == idx);
        if !has_var {
            let start = if idx < obs_names.len() {
                input
                    .s
                    .get(idx)
                    .and_then(|row| row.get(idx))
                    .copied()
                    .unwrap_or(1.0)
            } else {
                1.0
            };
            let coef = if std_lv && latent_names.contains(name) {
                CoefSpec::Fixed(1.0)
            } else {
                CoefSpec::Start(start)
            };
            slots.push(make_slot(name, "~~", name, MatKind::S, idx, idx, coef));
        }
    }

    // Add default covariances among exogenous latent variables
    let exog_latents: Vec<String> = latent_names
        .iter()
        .filter(|l| !latent_has_incoming.get(*l).copied().unwrap_or(false))
        .cloned()
        .collect();
    for i in 0..exog_latents.len() {
        for j in (i + 1)..exog_latents.len() {
            let lhs = &exog_latents[i];
            let rhs = &exog_latents[j];
            let row = *name_to_idx.get(lhs).unwrap();
            let col = *name_to_idx.get(rhs).unwrap();
            let has_cov = slots.iter().any(|s| {
                s.kind == MatKind::S
                    && ((s.row == row && s.col == col) || (s.row == col && s.col == row))
            });
            if !has_cov {
                slots.push(
                    make_slot(lhs, "~~", rhs, MatKind::S, row, col, CoefSpec::Start(0.0))
                        .with_symmetric(true),
                );
            }
        }
    }

    let (free, label_map, param_key_map) = assign_free_params(&mut slots)?;

    let (s_obs, s_vec) = extract_obs_s(&input.s, &obs_names)?;
    let wls_v = input.wls_v.clone().unwrap_or_else(|| {
        if matches!(input.estimation, Estimation::Ml) {
            build_wls_v_ml_from_s(&s_obs)
        } else {
            build_wls_v_from_gamma(&input.v, input.estimation)
        }
    });
    let w_diag = build_w_diag(&wls_v, s_vec.len());

    Ok(SemModel {
        var_names,
        obs_names,
        latent_names,
        slots,
        free,
        s_obs,
        gamma: input.v.clone(),
        wls_v,
        s_vec,
        w_diag,
        constraints: spec.constraints.clone(),
        param_key_map,
        label_map,
        estimation: input.estimation,
    })
}

fn build_model_from_table(
    input: &SemInput,
    table: &[ParTableRow],
) -> Result<(SemModel, ModelSpec)> {
    let obs_names = input.names.clone();
    let obs_set: HashSet<String> = obs_names.iter().cloned().collect();

    let mut latent_names = Vec::new();
    let mut defines = Vec::new();

    for row in table {
        if row.op == ":=" {
            let expr = parse_expr(&row.rhs)
                .with_context(|| format!("parse defined parameter {}", row.lhs))?;
            defines.push(DefineLine {
                name: row.lhs.clone(),
                expr,
                expr_raw: row.rhs.clone(),
            });
            continue;
        }
        if row.op == "=~" && !obs_set.contains(&row.lhs) && !latent_names.contains(&row.lhs) {
            latent_names.push(row.lhs.clone());
        }
    }

    let mut var_names = obs_names.clone();
    for latent in &latent_names {
        if !var_names.contains(latent) {
            var_names.push(latent.clone());
        }
    }

    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in var_names.iter().enumerate() {
        name_to_idx.insert(name.clone(), i);
    }

    let mut slots: Vec<ParamSlot> = Vec::new();

    for row in table {
        match row.op.as_str() {
            "=~" => {
                let lhs = row.lhs.clone();
                let rhs = row.rhs.clone();
                if rhs == "1" {
                    continue;
                }
                let row_idx = *name_to_idx.get(&rhs).context("unknown observed variable")?;
                let col_idx = *name_to_idx.get(&lhs).context("unknown latent variable")?;
                slots.push(make_slot_from_table(
                    row,
                    &lhs,
                    "=~",
                    &rhs,
                    MatKind::A,
                    row_idx,
                    col_idx,
                ));
            }
            "~" => {
                let lhs = row.lhs.clone();
                let rhs = row.rhs.clone();
                if rhs == "1" {
                    continue;
                }
                let row_idx = *name_to_idx.get(&lhs).context("unknown variable")?;
                let col_idx = *name_to_idx.get(&rhs).context("unknown variable")?;
                slots.push(make_slot_from_table(
                    row,
                    &lhs,
                    "~",
                    &rhs,
                    MatKind::A,
                    row_idx,
                    col_idx,
                ));
            }
            "~~" => {
                let lhs = row.lhs.clone();
                let rhs = row.rhs.clone();
                if rhs == "1" {
                    continue;
                }
                let row_idx = *name_to_idx.get(&lhs).context("unknown variable")?;
                let col_idx = *name_to_idx.get(&rhs).context("unknown variable")?;
                let symmetric = row_idx != col_idx;
                slots.push(
                    make_slot_from_table(row, &lhs, "~~", &rhs, MatKind::S, row_idx, col_idx)
                        .with_symmetric(symmetric),
                );
            }
            ":=" => {}
            _ => {}
        }
    }

    let (free, label_map, param_key_map) = assign_free_params_from_table(&mut slots)?;

    let (s_obs, s_vec) = extract_obs_s(&input.s, &obs_names)?;
    let wls_v = input.wls_v.clone().unwrap_or_else(|| {
        if matches!(input.estimation, Estimation::Ml) {
            build_wls_v_ml_from_s(&s_obs)
        } else {
            build_wls_v_from_gamma(&input.v, input.estimation)
        }
    });
    let w_diag = build_w_diag(&wls_v, s_vec.len());

    let model = SemModel {
        var_names,
        obs_names,
        latent_names,
        slots,
        free,
        s_obs,
        gamma: input.v.clone(),
        wls_v,
        s_vec,
        w_diag,
        constraints: Vec::new(),
        param_key_map,
        label_map,
        estimation: input.estimation,
    };

    let spec = ModelSpec {
        lines: Vec::new(),
        constraints: Vec::new(),
        defines,
    };

    Ok((model, spec))
}

fn make_slot(
    lhs: &str,
    op: &str,
    rhs: &str,
    kind: MatKind,
    row: usize,
    col: usize,
    coef: CoefSpec,
) -> ParamSlot {
    let (fixed, label, start) = match coef {
        CoefSpec::Fixed(v) => (Some(v), None, None),
        CoefSpec::Label(l) => (None, Some(l), None),
        CoefSpec::Start(v) => (None, None, Some(v)),
        CoefSpec::NA => (None, None, None),
        CoefSpec::None => (None, None, None),
    };
    ParamSlot {
        lhs: lhs.to_string(),
        op: op.to_string(),
        rhs: rhs.to_string(),
        kind,
        row,
        col,
        symmetric: false,
        free_idx: 0,
        fixed,
        label,
        start,
    }
}

fn make_slot_from_table(
    row: &ParTableRow,
    lhs: &str,
    op: &str,
    rhs: &str,
    kind: MatKind,
    row_idx: usize,
    col_idx: usize,
) -> ParamSlot {
    let (fixed, start) = if row.free == 0 {
        (Some(row.ustart), None)
    } else {
        (None, Some(row.ustart))
    };
    ParamSlot {
        lhs: lhs.to_string(),
        op: op.to_string(),
        rhs: rhs.to_string(),
        kind,
        row: row_idx,
        col: col_idx,
        symmetric: false,
        free_idx: row.free,
        fixed,
        label: row.label.clone(),
        start,
    }
}

trait SymmetricSlot {
    fn with_symmetric(self, symmetric: bool) -> Self;
}

impl SymmetricSlot for ParamSlot {
    fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
}

type FreeMaps = (
    Vec<FreeParam>,
    HashMap<String, usize>,
    HashMap<String, usize>,
);

fn assign_free_params(slots: &mut [ParamSlot]) -> Result<FreeMaps> {
    let mut free = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();
    let mut param_key_map: HashMap<String, usize> = HashMap::new();

    for slot in slots.iter_mut() {
        let key = format!("{}{}{}", slot.lhs, slot.op, slot.rhs);
        if slot.fixed.is_some() {
            slot.free_idx = 0;
        } else if let Some(label) = &slot.label {
            if let Some(&idx) = label_map.get(label) {
                slot.free_idx = idx;
            } else {
                let idx = free.len() + 1;
                slot.free_idx = idx;
                label_map.insert(label.clone(), idx);
                free.push(FreeParam {
                    label: Some(label.clone()),
                    start: slot.start.unwrap_or(default_start(slot)),
                    lower: None,
                    upper: None,
                });
            }
        } else {
            let idx = free.len() + 1;
            slot.free_idx = idx;
            free.push(FreeParam {
                label: None,
                start: slot.start.unwrap_or(default_start(slot)),
                lower: None,
                upper: None,
            });
        }
        if slot.free_idx > 0 {
            param_key_map.insert(key, slot.free_idx);
        }
    }

    Ok((free, label_map, param_key_map))
}

fn assign_free_params_from_table(slots: &mut [ParamSlot]) -> Result<FreeMaps> {
    let mut free: Vec<FreeParam> = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();
    let mut param_key_map: HashMap<String, usize> = HashMap::new();

    let max_free = slots.iter().map(|s| s.free_idx).max().unwrap_or(0);
    free.resize_with(max_free, || FreeParam {
        label: None,
        start: 0.0,
        lower: None,
        upper: None,
    });

    for slot in slots.iter_mut() {
        if slot.free_idx > 0 {
            let idx = slot.free_idx - 1;
            if free[idx].start == 0.0 && slot.start.is_some() {
                free[idx].start = slot.start.unwrap_or(0.0);
            }
            if let Some(label) = &slot.label {
                label_map.entry(label.clone()).or_insert(slot.free_idx);
            }
            let key = format!("{}{}{}", slot.lhs, slot.op, slot.rhs);
            param_key_map.insert(key, slot.free_idx);
        }
    }

    Ok((free, label_map, param_key_map))
}

fn default_start(slot: &ParamSlot) -> f64 {
    match slot.op.as_str() {
        "~~" => {
            if slot.row == slot.col {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.1,
    }
}

fn extract_obs_s(s: &Matrix, obs_names: &[String]) -> Result<(Matrix, Vec<f64>)> {
    let k = obs_names.len();
    if s.len() < k {
        return Err(anyhow::anyhow!("S matrix smaller than observed variables"));
    }
    let mut s_obs = vec![vec![0.0; k]; k];
    for (i, row) in s_obs.iter_mut().enumerate().take(k) {
        for (j, cell) in row.iter_mut().enumerate().take(k) {
            *cell = s[i][j];
        }
    }
    let mut vec = Vec::with_capacity(k * (k + 1) / 2);
    for (j, _) in s_obs.iter().enumerate().take(k) {
        for row in s_obs.iter().skip(j).take(k - j) {
            vec.push(row[j]);
        }
    }
    Ok((s_obs, vec))
}

fn build_wls_v_from_gamma(gamma: &Matrix, estimation: crate::types::Estimation) -> Matrix {
    match estimation {
        crate::types::Estimation::Dwls => build_wls_v_diag(gamma),
        crate::types::Estimation::Ml => build_wls_v_full(gamma),
    }
}

fn build_wls_v_ml_from_s(s_obs: &Matrix) -> Matrix {
    h1_expected_information(s_obs, false)
}

fn build_wls_v_diag(gamma: &Matrix) -> Matrix {
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

fn build_wls_v_full(gamma: &Matrix) -> Matrix {
    if let Ok(arr) = crate::implied::to_array2(gamma)
        && let Ok(inv) = crate::linalg::inverse_from_matrix(&arr, Some(1e-12))
    {
        return crate::implied::from_array2(&inv);
    }
    build_wls_v_diag(gamma)
}

fn h1_expected_information(sample_cov: &Matrix, meanstructure: bool) -> Matrix {
    let s_inv = symmetric_inverse(sample_cov);
    let i22 = h1_expected_information_cov(&s_inv);
    if meanstructure {
        block_diag(&s_inv, &i22)
    } else {
        i22
    }
}

fn h1_expected_information_cov(s_inv: &Matrix) -> Matrix {
    let kron = kron_matrix(s_inv, s_inv);
    let mut out = duplication_pre_post(&kron, true);
    for row in out.iter_mut() {
        for val in row.iter_mut() {
            *val *= 0.5;
        }
    }
    out
}

fn symmetric_inverse(matrix: &Matrix) -> Matrix {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let m = matrix.first().map(|r| r.len()).unwrap_or(0);
    if n != m {
        return vec![vec![f64::NAN; m]; n];
    }

    let mut zero_idx = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        let row_sum: f64 = row.iter().copied().sum();
        let col_sum: f64 = matrix.iter().map(|r| r[i]).sum();
        if row_sum == 0.0 && col_sum == 0.0 && row[i] == 0.0 {
            zero_idx.push(i);
        }
    }

    let (work, kept): (Matrix, Vec<usize>) = if zero_idx.is_empty() {
        (matrix.clone(), (0..n).collect())
    } else {
        let mut kept = Vec::new();
        for i in 0..n {
            if !zero_idx.contains(&i) {
                kept.push(i);
            }
        }
        let mut work = vec![vec![0.0; kept.len()]; kept.len()];
        for (ri, &i) in kept.iter().enumerate() {
            for (cj, &j) in kept.iter().enumerate() {
                work[ri][cj] = matrix[i][j];
            }
        }
        (work, kept)
    };

    let p = work.len();
    let inv_work = if p == 0 {
        vec![]
    } else if p == 1 {
        let tmp = work[0][0];
        vec![vec![1.0 / tmp]]
    } else if p == 2 {
        let a11 = work[0][0];
        let a12 = work[0][1];
        let a21 = work[1][0];
        let a22 = work[1][1];
        let tmp = a11 * a22 - a12 * a21;
        if tmp == 0.0 {
            vec![
                vec![f64::INFINITY, f64::INFINITY],
                vec![f64::INFINITY, f64::INFINITY],
            ]
        } else {
            vec![vec![a22 / tmp, -a21 / tmp], vec![-a12 / tmp, a11 / tmp]]
        }
    } else if let Ok(arr) = crate::implied::to_array2(&work)
        && let Ok(inv) = crate::linalg::inverse_from_matrix(&arr, None)
    {
        crate::implied::from_array2(&inv)
    } else {
        return build_wls_v_diag(&work);
    };

    if zero_idx.is_empty() {
        return inv_work;
    }
    let mut out = matrix.clone();
    for (ri, &i) in kept.iter().enumerate() {
        for (cj, &j) in kept.iter().enumerate() {
            out[i][j] = inv_work[ri][cj];
        }
    }
    out
}

fn kron_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    let m = a.len();
    let n = a.first().map(|r| r.len()).unwrap_or(0);
    let p = b.len();
    let q = b.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![vec![0.0; n * q]; m * p];
    for i in 0..m {
        for j in 0..n {
            let aij = a[i][j];
            for ii in 0..p {
                for jj in 0..q {
                    out[i * p + ii][j * q + jj] = aij * b[ii][jj];
                }
            }
        }
    }
    out
}

fn duplication_pre_post(a: &Matrix, diagonal: bool) -> Matrix {
    let n2 = a.len();
    if n2 == 0 {
        return vec![];
    }
    let n = (n2 as f64).sqrt() as usize;
    let idx1 = vech_idx(n, diagonal);
    let idx2 = vechru_idx(n, diagonal);
    let pstar = idx1.len();
    let mut out = vec![vec![0.0; n2]; pstar];
    for (r, out_row) in out.iter_mut().enumerate() {
        let row1 = &a[idx1[r]];
        let row2 = &a[idx2[r]];
        for (c, cell) in out_row.iter_mut().enumerate() {
            *cell = row1[c] + row2[c];
        }
    }
    let idx2_set: std::collections::HashSet<usize> = idx2.iter().copied().collect();
    let mut u = Vec::new();
    for (pos, idx) in idx1.iter().enumerate() {
        if idx2_set.contains(idx) {
            u.push(pos);
        }
    }
    for &pos in &u {
        for cell in out[pos].iter_mut() {
            *cell *= 0.5;
        }
    }
    let mut out2 = vec![vec![0.0; pstar]; pstar];
    for (r, row) in out2.iter_mut().enumerate() {
        let out_row = &out[r];
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = out_row[idx1[c]] + out_row[idx2[c]];
        }
    }
    for &pos in &u {
        for row in out2.iter_mut() {
            row[pos] *= 0.5;
        }
    }
    out2
}

fn vech_idx(n: usize, diagonal: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for j in 0..n {
        let start = if diagonal { j } else { j + 1 };
        for i in start..n {
            out.push(i + j * n);
        }
    }
    out
}

fn vechru_idx(n: usize, diagonal: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..n {
        let end = if diagonal { i } else { i.saturating_sub(1) };
        for j in 0..=end {
            out.push(i * n + j);
        }
    }
    out
}

fn block_diag(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let m = b.len();
    let mut out = vec![vec![0.0; n + m]; n + m];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j];
        }
    }
    for i in 0..m {
        for j in 0..m {
            out[n + i][n + j] = b[i][j];
        }
    }
    out
}

fn build_w_diag(wls_v: &Matrix, len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (i, out_val) in out.iter_mut().enumerate().take(len) {
        if let Some(row) = wls_v.get(i) {
            let val = row.get(i).copied().unwrap_or(f64::NAN);
            if val.is_finite() && val != 0.0 {
                *out_val = val;
            }
        }
    }
    out
}

impl SemModel {
    pub fn theta_start(&self) -> Vec<f64> {
        self.free.iter().map(|f| f.start).collect()
    }

    pub fn bounds(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut any_lower = false;
        let mut any_upper = false;
        for f in &self.free {
            if let Some(l) = f.lower {
                lower.push(l);
                any_lower = true;
            } else {
                lower.push(f64::NEG_INFINITY);
            }
            if let Some(u) = f.upper {
                upper.push(u);
                any_upper = true;
            } else {
                upper.push(f64::INFINITY);
            }
        }
        let lower = if any_lower { Some(lower) } else { None };
        let upper = if any_upper { Some(upper) } else { None };
        (lower, upper)
    }

    pub fn apply_constraints(&mut self) {
        for constraint in &self.constraints {
            let idx = if let Some(&i) = self.label_map.get(&constraint.target) {
                Some(i)
            } else if let Some(&i) = self.param_key_map.get(&constraint.target) {
                Some(i)
            } else {
                None
            };
            if let Some(idx) = idx
                && let Some(free) = self.free.get_mut(idx - 1)
            {
                match constraint.op {
                    crate::parser::ConstraintOp::Gt | crate::parser::ConstraintOp::Ge => {
                        free.lower = Some(constraint.value);
                    }
                    crate::parser::ConstraintOp::Lt | crate::parser::ConstraintOp::Le => {
                        free.upper = Some(constraint.value);
                    }
                    crate::parser::ConstraintOp::Eq => {
                        free.lower = Some(constraint.value);
                        free.upper = Some(constraint.value);
                    }
                }
            }
        }
    }

    pub fn build_matrices(&self, theta: &[f64]) -> Result<(Matrix, Matrix)> {
        let n = self.var_names.len();
        let mut a = vec![vec![0.0; n]; n];
        let mut s = vec![vec![0.0; n]; n];
        for slot in &self.slots {
            let value = if slot.free_idx > 0 {
                theta.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
            } else {
                slot.fixed.unwrap_or(f64::NAN)
            };
            match slot.kind {
                MatKind::A => {
                    a[slot.row][slot.col] = value;
                }
                MatKind::S => {
                    s[slot.row][slot.col] = value;
                    if slot.symmetric {
                        s[slot.col][slot.row] = value;
                    }
                }
            }
        }
        Ok((a, s))
    }
}
