use anyhow::{Context, Result};
use chumsky::prelude::*;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOp {
    Measure,
    Regress,
    Cov,
}

#[derive(Debug, Clone)]
pub enum CoefSpec {
    Fixed(f64),
    Label(String),
    Start(f64),
    NA,
    None,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub coef: CoefSpec,
    pub var: String,
}

#[derive(Debug, Clone)]
pub struct Line {
    pub lhs: String,
    pub op: ModelOp,
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub target: String,
    pub op: ConstraintOp,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct DefineLine {
    pub name: String,
    pub expr: Expr,
    pub expr_raw: String,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub lines: Vec<Line>,
    pub constraints: Vec<Constraint>,
    pub defines: Vec<DefineLine>,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    Var(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Func(String, Vec<Expr>),
}

enum ParsedLine {
    Model(Line),
    Constraint(Constraint),
}

fn any_char<'a>()
-> impl Parser<'a, &'a str, char, chumsky::extra::Err<chumsky::error::Simple<'a, char>>> + Copy {
    any()
}

fn ident_parser<'a>()
-> impl Parser<'a, &'a str, String, chumsky::extra::Err<chumsky::error::Simple<'a, char>>> + Clone {
    let ident_start =
        any_char().filter(|c: &char| c.is_ascii_alphabetic() || *c == '_' || *c == '.');
    let ident_rest = any_char()
        .filter(|c: &char| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
        .repeated()
        .collect::<String>();
    ident_start
        .then(ident_rest)
        .map(|(first, rest)| {
            let mut s = String::new();
            s.push(first);
            s.push_str(&rest);
            s
        })
        .padded()
}

pub fn parse_model(model: &str) -> Result<ModelSpec> {
    let mut lines = Vec::new();
    let mut constraints = Vec::new();
    let mut defines = Vec::new();

    for raw_line in model.lines() {
        let stripped = strip_comments(raw_line);
        for segment in stripped.split(';') {
            let line = segment.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((lhs, rhs)) = line.split_once(":=") {
                let name = lhs.trim().to_string();
                let expr_raw = rhs.trim().to_string();
                let expr = parse_expr(&expr_raw)
                    .with_context(|| format!("parse defined parameter {name}"))?;
                defines.push(DefineLine {
                    name,
                    expr,
                    expr_raw,
                });
                continue;
            }
            match parse_line_with_chumsky(line)? {
                ParsedLine::Model(line) => lines.push(line),
                ParsedLine::Constraint(constraint) => constraints.push(constraint),
            }
        }
    }

    Ok(ModelSpec {
        lines,
        constraints,
        defines,
    })
}

fn strip_comments(line: &str) -> String {
    let mut out = line.to_string();
    if let Some(idx) = out.find('#') {
        out.truncate(idx);
    }
    if let Some(idx) = out.find("//") {
        out.truncate(idx);
    }
    out
}

fn parse_line_with_chumsky(line: &str) -> Result<ParsedLine> {
    let ident = ident_parser();

    let sign = just('-').or(just('+')).or_not();
    let digits = any_char()
        .filter(|c: &char| c.is_ascii_digit())
        .repeated()
        .at_least(1)
        .collect::<String>();
    let int = digits;
    let frac = just('.').then(digits).or_not();
    let exp = just('e')
        .or(just('E'))
        .then(just('-').or(just('+')).or_not())
        .then(digits)
        .or_not();

    let number = sign
        .then(int)
        .then(frac)
        .then(exp)
        .map(|(((sign, int), frac), exp)| {
            let mut s = String::new();
            if let Some(sign) = sign {
                s.push(sign);
            }
            s.push_str(&int);
            if let Some((dot, frac)) = frac {
                s.push(dot);
                s.push_str(&frac);
            }
            if let Some(((e, sign), digits)) = exp {
                s.push(e);
                if let Some(sign) = sign {
                    s.push(sign);
                }
                s.push_str(&digits);
            }
            s
        })
        .map(|s| f64::from_str(&s).unwrap_or(f64::NAN))
        .padded();

    let coef = choice((
        just("NA").to(CoefSpec::NA),
        just("start")
            .padded()
            .ignore_then(just('(').padded())
            .ignore_then(number)
            .then_ignore(just(')').padded())
            .map(CoefSpec::Start),
        number.map(CoefSpec::Fixed),
        ident.clone().map(CoefSpec::Label),
    ));

    let term = coef
        .then_ignore(just('*').padded())
        .then(ident.clone())
        .map(|(coef, var)| Term { coef, var })
        .or(ident.clone().map(|var| Term {
            coef: CoefSpec::None,
            var,
        }));

    let terms = term
        .separated_by(just('+').padded())
        .at_least(1)
        .collect::<Vec<_>>();

    let op = choice((
        just("=~").to(ModelOp::Measure),
        just("~~").to(ModelOp::Cov),
        just('~').to(ModelOp::Regress),
    ))
    .padded();

    let model_line = ident
        .clone()
        .then(op)
        .then(terms)
        .map(|((lhs, op), terms)| ParsedLine::Model(Line { lhs, op, terms }));

    let constraint_op = choice((
        just("<=").to(ConstraintOp::Le),
        just(">=").to(ConstraintOp::Ge),
        just("==").to(ConstraintOp::Eq),
        just('<').to(ConstraintOp::Lt),
        just('>').to(ConstraintOp::Gt),
    ))
    .padded();

    let constraint = ident
        .then(constraint_op)
        .then(number)
        .map(|((target, op), value)| ParsedLine::Constraint(Constraint { target, op, value }));

    let parser = model_line.or(constraint).then_ignore(end());
    parser.parse(line).into_result().map_err(|errs| {
        let msg = errs
            .into_iter()
            .map(|e: chumsky::error::Simple<char>| {
                format!("parse error at {:?}: found {:?}", e.span(), e.found())
            })
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::anyhow!("Failed to parse model line: {line}. {msg}")
    })
}

// Expression parsing for defined parameters (:=)
pub fn parse_expr(expr: &str) -> Result<Expr> {
    use chumsky::pratt::*;

    let ident = ident_parser();

    let digits = any_char()
        .filter(|c: &char| c.is_ascii_digit())
        .repeated()
        .at_least(1)
        .collect::<String>();

    let frac = just('.').then(digits).or_not();
    let exp = just('e')
        .or(just('E'))
        .then(just('-').or(just('+')).or_not())
        .then(digits)
        .or_not();

    let number = digits
        .then(frac)
        .then(exp)
        .map(|((int, frac), exp)| {
            let mut s = int;
            if let Some((dot, frac)) = frac {
                s.push(dot);
                s.push_str(&frac);
            }
            if let Some(((e, sign), digits)) = exp {
                s.push(e);
                if let Some(sign) = sign {
                    s.push(sign);
                }
                s.push_str(&digits);
            }
            s
        })
        .map(|s| f64::from_str(&s).unwrap_or(f64::NAN))
        .padded();

    let expr_parser = recursive(|expr| {
        let func = ident
            .clone()
            .then(
                expr.clone()
                    .separated_by(just(',').padded())
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just('(').padded(), just(')').padded()),
            )
            .map(|(name, args)| Expr::Func(name, args));

        let atom = choice((
            func,
            number.map(Expr::Number),
            ident.clone().map(Expr::Var),
            expr.clone()
                .delimited_by(just('(').padded(), just(')').padded()),
        ))
        .padded();

        atom.pratt((
            prefix(3, just('-').padded(), |_, rhs, _| Expr::Neg(Box::new(rhs))),
            infix(right(2), just('^').padded(), |lhs, _, rhs, _| {
                Expr::Pow(Box::new(lhs), Box::new(rhs))
            }),
            infix(left(1), just('*').padded(), |lhs, _, rhs, _| {
                Expr::Mul(Box::new(lhs), Box::new(rhs))
            }),
            infix(left(1), just('/').padded(), |lhs, _, rhs, _| {
                Expr::Div(Box::new(lhs), Box::new(rhs))
            }),
            infix(left(0), just('+').padded(), |lhs, _, rhs, _| {
                Expr::Add(Box::new(lhs), Box::new(rhs))
            }),
            infix(left(0), just('-').padded(), |lhs, _, rhs, _| {
                Expr::Sub(Box::new(lhs), Box::new(rhs))
            }),
        ))
    });

    expr_parser.parse(expr).into_result().map_err(|errs| {
        let msg = errs
            .into_iter()
            .map(|e: chumsky::error::Simple<char>| {
                format!("parse error at {:?}: found {:?}", e.span(), e.found())
            })
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::anyhow!("Failed to parse defined parameter expression: {expr}. {msg}")
    })
}

pub fn eval_expr(expr: &Expr, vars: &std::collections::HashMap<String, f64>) -> Result<f64> {
    Ok(match expr {
        Expr::Number(v) => *v,
        Expr::Var(name) => *vars.get(name).unwrap_or(&f64::NAN),
        Expr::Neg(e) => -eval_expr(e, vars)?,
        Expr::Add(a, b) => eval_expr(a, vars)? + eval_expr(b, vars)?,
        Expr::Sub(a, b) => eval_expr(a, vars)? - eval_expr(b, vars)?,
        Expr::Mul(a, b) => eval_expr(a, vars)? * eval_expr(b, vars)?,
        Expr::Div(a, b) => eval_expr(a, vars)? / eval_expr(b, vars)?,
        Expr::Pow(a, b) => eval_expr(a, vars)?.powf(eval_expr(b, vars)?),
        Expr::Func(name, args) => {
            let mut vals = Vec::new();
            for a in args {
                vals.push(eval_expr(a, vars)?);
            }
            match name.as_str() {
                "exp" => vals.first().copied().unwrap_or(f64::NAN).exp(),
                "log" => vals.first().copied().unwrap_or(f64::NAN).ln(),
                "sqrt" => vals.first().copied().unwrap_or(f64::NAN).sqrt(),
                "pow" => {
                    if vals.len() >= 2 {
                        vals[0].powf(vals[1])
                    } else {
                        f64::NAN
                    }
                }
                _ => f64::NAN,
            }
        }
    })
}
