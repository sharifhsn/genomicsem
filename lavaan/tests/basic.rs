use lavaan::implied::{implied_observed, vech_col_major};
use lavaan::model::SemModel;
use lavaan::parser::parse_model;
use lavaan::types::{Estimation, SemInput};

#[test]
fn parse_model_smoke() {
    let model = "f1 =~ y1 + 0.5*y2\n y2 ~ f1\n y1 ~~ y2";
    let spec = parse_model(model).expect("parse model");
    assert_eq!(spec.lines.len(), 3);
    assert!(spec.constraints.is_empty());
}

#[test]
fn vech_column_major() {
    let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let vech = vech_col_major(&matrix);
    assert_eq!(vech, vec![1.0, 3.0, 4.0]);
}

#[test]
fn implied_covariance_singleton() {
    let input = SemInput {
        s: vec![vec![2.0]],
        v: vec![vec![1.0]],
        wls_v: None,
        model: "".to_string(),
        model_table: None,
        estimation: Estimation::Dwls,
        toler: None,
        std_lv: false,
        fix_measurement: false,
        q_snp: false,
        names: vec!["y1".to_string()],
        n_obs: None,
        optim_dx_tol: None,
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    };
    let spec = parse_model(&input.model).expect("parse model");
    let model = SemModel::build_model(&input, &spec).expect("build model");
    let theta = model.theta_start();
    let sigma = implied_observed(&model, &theta).expect("implied");
    assert_eq!(sigma.len(), 1);
    assert!((sigma[0][0] - 2.0).abs() < 1e-8);
}
