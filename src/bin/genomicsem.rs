use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use polars::prelude::*;

use genomicsem::gwas::{
    CommonFactorGwasConfig, UserGwasConfig, commonfactor_gwas, commonfactor_gwas_output_table,
    user_gwas, user_gwas_output_table,
};
use genomicsem::hdl::{HdlConfig, HdlMethod, hdl};
use genomicsem::io::{
    read_ldsc_json, read_matrix_file, read_s_ldsc_json, read_table, read_vector_file,
    write_dataframe, write_ldsc_json, write_matrix, write_s_ldsc_json, write_scalar, write_vector,
};
use genomicsem::ldsc::{ChromosomeSelect, LdscConfig, ldsc};
use genomicsem::logging::init_tracing;
use genomicsem::munge::{MungeConfig, munge};
use genomicsem::plot_utils::{ensure_plots_dir, plot_path};
use genomicsem::post_ldsc::{
    SubSvType, SummaryGlsBandsOutput, SummaryGlsOutput, index_s_from_matrix, local_srmd, pa_ldsc,
    sub_sv_from_matrices, summary_gls, summary_gls_bands,
};
use genomicsem::qtrait::{QTraitConfig, qtrait};
use genomicsem::sem::{
    commonfactor, commonfactor_output_tables, usermodel, usermodel_output_tables,
};
use genomicsem::sim_ldsc::{
    CovInput, MatrixOrScalar, NInput, SimLdscConfig, VecOrScalar, sim_ldsc,
};
use genomicsem::stratified::{EnrichConfig, SLdscConfig, enrich, s_ldsc};
use genomicsem::sumstats::{SumstatsConfig, sumstats};
use genomicsem::twas::{ReadFusionConfig, read_fusion};
use genomicsem::types::{Estimation, GenomicControl, Matrix, SumstatsTable};
use plotly::common::color::{NamedColor, Rgb, Rgba};
use plotly::common::{DashType, Fill, Line, Marker, MarkerSymbol, Mode};
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};

#[derive(Parser)]
#[command(name = "genomicsem")]
#[command(about = "GenomicSEM Rust port", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Munge {
        #[arg(long, required = true)]
        files: Vec<PathBuf>,
        #[arg(long, required = true)]
        hm3: PathBuf,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long)]
        n: Option<String>,
        #[arg(long, default_value_t = 0.9)]
        info_filter: f64,
        #[arg(long, default_value_t = 0.01)]
        maf_filter: f64,
        #[arg(long)]
        overwrite: bool,
        #[arg(long)]
        log_name: Option<String>,
    },
    Sumstats {
        #[arg(long, required = true)]
        files: Vec<PathBuf>,
        #[arg(long, required = true)]
        reference: PathBuf,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long, required = true)]
        se_logit: String,
        #[arg(long)]
        ols: Option<String>,
        #[arg(long)]
        linprob: Option<String>,
        #[arg(long)]
        n: Option<String>,
        #[arg(long)]
        betas: Option<String>,
        #[arg(long, default_value_t = 0.6)]
        info_filter: f64,
        #[arg(long, default_value_t = 0.01)]
        maf_filter: f64,
        #[arg(long)]
        keep_indel: bool,
        #[arg(long)]
        ambig: bool,
        #[arg(long)]
        direct_filter: bool,
        #[arg(long)]
        parallel: bool,
        #[arg(long)]
        cores: Option<usize>,
        #[arg(long)]
        log_name: Option<String>,
    },
    Ldsc {
        #[arg(long, required = true)]
        traits: Vec<PathBuf>,
        #[arg(long)]
        sample_prev: Option<String>,
        #[arg(long)]
        population_prev: Option<String>,
        #[arg(long, required = true)]
        ld: PathBuf,
        #[arg(long, required = true)]
        wld: PathBuf,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long)]
        sep_weights: bool,
        #[arg(long, default_value_t = 22)]
        chr: u8,
        #[arg(long, default_value_t = 200)]
        n_blocks: usize,
        #[arg(long)]
        ldsc_log: Option<String>,
        #[arg(long)]
        stand: bool,
        #[arg(long)]
        select: Option<String>,
        #[arg(long)]
        chisq_max: Option<f64>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    SLdsc {
        #[arg(long, required = true)]
        traits: Vec<PathBuf>,
        #[arg(long)]
        sample_prev: Option<String>,
        #[arg(long)]
        population_prev: Option<String>,
        #[arg(long, required = true)]
        ld: Vec<PathBuf>,
        #[arg(long, required = true)]
        wld: PathBuf,
        #[arg(long, required = true)]
        frq: PathBuf,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long, default_value_t = 200)]
        n_blocks: usize,
        #[arg(long)]
        ldsc_log: Option<PathBuf>,
        #[arg(long)]
        exclude_cont: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Enrich {
        #[arg(long, required = true)]
        s_covstruc: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long, required = true)]
        params: String,
        #[arg(long, default_value = "regressions")]
        fix: String,
        #[arg(long)]
        std_lv: bool,
        #[arg(long, default_value_t = true)]
        rm_flank: bool,
        #[arg(long)]
        tau: bool,
        #[arg(long, default_value_t = true)]
        base: bool,
        #[arg(long)]
        toler: Option<f64>,
        #[arg(long)]
        fixparam: Option<String>,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    CommonfactorGwas {
        #[arg(long, required = true)]
        covstruc: PathBuf,
        #[arg(long, required = true)]
        snps: PathBuf,
        #[arg(long, default_value = "DWLS")]
        estimation: String,
        #[arg(long)]
        cores: Option<usize>,
        #[arg(long)]
        toler: Option<f64>,
        #[arg(long)]
        snp_se: Option<f64>,
        #[arg(long, default_value_t = true)]
        parallel: bool,
        #[arg(long, default_value = "standard")]
        gc: String,
        #[arg(long)]
        mpi: bool,
        #[arg(long)]
        twas: bool,
        #[arg(long)]
        smooth_check: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    UserGwas {
        #[arg(long, required = true)]
        covstruc: PathBuf,
        #[arg(long, required = true)]
        snps: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long, default_value = "DWLS")]
        estimation: String,
        #[arg(long, default_value_t = true)]
        printwarn: bool,
        #[arg(long)]
        sub: Option<String>,
        #[arg(long)]
        cores: Option<usize>,
        #[arg(long)]
        toler: Option<f64>,
        #[arg(long)]
        snp_se: Option<f64>,
        #[arg(long, default_value_t = true)]
        parallel: bool,
        #[arg(long, default_value = "standard")]
        gc: String,
        #[arg(long)]
        mpi: bool,
        #[arg(long)]
        smooth_check: bool,
        #[arg(long)]
        twas: bool,
        #[arg(long)]
        std_lv: bool,
        #[arg(long, default_value_t = true)]
        fix_measurement: bool,
        #[arg(long)]
        q_snp: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Commonfactor {
        #[arg(long, required = true)]
        covstruc: PathBuf,
        #[arg(long, default_value = "DWLS")]
        estimation: String,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    Usermodel {
        #[arg(long, required = true)]
        covstruc: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long, default_value = "DWLS")]
        estimation: String,
        #[arg(long)]
        std_lv: bool,
        #[arg(long, default_value_t = true)]
        cfi_calc: bool,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    ReadFusion {
        #[arg(long, required = true)]
        files: Vec<PathBuf>,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long)]
        binary: Option<String>,
        #[arg(long)]
        n: Option<String>,
        #[arg(long)]
        perm: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Hdl {
        #[arg(long, required = true)]
        traits: Vec<PathBuf>,
        #[arg(long)]
        sample_prev: Option<String>,
        #[arg(long)]
        population_prev: Option<String>,
        #[arg(long, required = true)]
        ld_path: PathBuf,
        #[arg(long, default_value_t = 335265)]
        n_ref: usize,
        #[arg(long)]
        trait_names: Option<String>,
        #[arg(long, default_value = "piecewise")]
        method: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    PaLdsc {
        #[arg(long, required = true)]
        s: PathBuf,
        #[arg(long, required = true)]
        v: PathBuf,
        #[arg(long, default_value_t = 500)]
        r: usize,
        #[arg(long, default_value_t = 0.95)]
        p: f64,
        #[arg(long)]
        diag: bool,
        #[arg(long)]
        fa: bool,
        #[arg(long, default_value_t = 1)]
        nfactors: usize,
        #[arg(long)]
        save_plots: bool,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    IndexS {
        #[arg(long, required = true)]
        s: PathBuf,
        #[arg(long)]
        include_diag: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    SubSv {
        #[arg(long, required = true)]
        s: PathBuf,
        #[arg(long, required = true)]
        v: PathBuf,
        #[arg(long, required = true)]
        index: String,
        #[arg(long, default_value = "S")]
        ty: String,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    SummaryGls {
        #[arg(long, required = true)]
        y: PathBuf,
        #[arg(long, required = true)]
        v: PathBuf,
        #[arg(long, required = true)]
        predictors: PathBuf,
        #[arg(long)]
        intercept: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    SummaryGlsBands {
        #[arg(long, required = true)]
        y: PathBuf,
        #[arg(long, required = true)]
        v: PathBuf,
        #[arg(long, required = true)]
        predictors: PathBuf,
        #[arg(long)]
        controlvars: Option<PathBuf>,
        #[arg(long, default_value_t = 20)]
        intervals: usize,
        #[arg(long)]
        intercept: bool,
        #[arg(long)]
        quad: bool,
        #[arg(long, default_value_t = true)]
        bands: bool,
        #[arg(long)]
        no_bands: bool,
        #[arg(long, default_value_t = 1.0)]
        band_size: f64,
        #[arg(long, default_value_t = true)]
        save_plots: bool,
        #[arg(long, default_value = "")]
        xlab: String,
        #[arg(long, default_value = "")]
        ylab: String,
        #[arg(long, default_value = "")]
        title: String,
        #[arg(long, default_value = "-2,2")]
        xcoords: String,
        #[arg(long, default_value = "0,1")]
        ycoords: String,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    LocalSrmd {
        #[arg(long, required = true)]
        unconstrained: PathBuf,
        #[arg(long, required = true)]
        constrained: PathBuf,
        #[arg(long, required = true)]
        lhsvar: PathBuf,
        #[arg(long, required = true)]
        rhsvar: PathBuf,
        #[arg(long)]
        output_prefix: Option<String>,
    },
    SimLdsc {
        #[arg(long, required = true)]
        covmat: PathBuf,
        #[arg(long, required = true)]
        n: String,
        #[arg(long)]
        ld: PathBuf,
        #[arg(long)]
        r_pheno: Option<String>,
        #[arg(long)]
        intercepts: Option<String>,
        #[arg(long, default_value_t = 0.99)]
        n_overlap: f64,
        #[arg(long, default_value_t = 1)]
        r: usize,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
        #[arg(long)]
        gzip_output: bool,
        #[arg(long)]
        parallel: bool,
        #[arg(long)]
        cores: Option<usize>,
    },
    QTrait {
        #[arg(long, required = true)]
        ldsc: PathBuf,
        #[arg(long, required = true)]
        indicators: String,
        #[arg(long, required = true)]
        traits: String,
        #[arg(long, default_value_t = 0.25)]
        mresid: f64,
        #[arg(long, default_value_t = 0.10)]
        mresid_threshold: f64,
        #[arg(long, default_value_t = 0.25)]
        lsrmr: f64,
        #[arg(long, default_value_t = 0.10)]
        lsrmr_threshold: f64,
        #[arg(long, default_value_t = true)]
        save_plots: bool,
        #[arg(long, default_value_t = true)]
        stdout: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Munge {
            files,
            hm3,
            trait_names,
            n,
            info_filter,
            maf_filter,
            overwrite,
            log_name,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let n = n.map(split_f64_list);
            let config = MungeConfig {
                files,
                hm3,
                trait_names,
                n,
                info_filter,
                maf_filter,
                column_names: Default::default(),
                parallel: false,
                cores: None,
                overwrite,
                log_name,
            };
            munge(&config)?;
        }
        Command::Sumstats {
            files,
            reference,
            trait_names,
            se_logit,
            ols,
            linprob,
            n,
            betas,
            info_filter,
            maf_filter,
            keep_indel,
            ambig,
            direct_filter,
            parallel,
            cores,
            log_name,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let se_logit = split_bool_list(&se_logit);
            let ols = ols.map(|v| split_bool_list(&v));
            let linprob = linprob.map(|v| split_bool_list(&v));
            let n = n.map(split_f64_list);
            let betas = betas.map(split_string_list);
            let config = SumstatsConfig {
                files,
                reference,
                trait_names,
                se_logit,
                ols,
                linprob,
                n,
                betas,
                info_filter,
                maf_filter,
                keep_indel,
                parallel,
                cores,
                ambig,
                direct_filter,
                column_names: Default::default(),
                log_name,
            };
            sumstats(&config)?;
        }
        Command::Ldsc {
            traits,
            sample_prev,
            population_prev,
            ld,
            wld,
            trait_names,
            sep_weights,
            chr,
            n_blocks,
            ldsc_log,
            stand,
            select,
            chisq_max,
            output,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let sample_prev = sample_prev.map(parse_optional_f64_list).unwrap_or_default();
            let population_prev = population_prev
                .map(parse_optional_f64_list)
                .unwrap_or_default();
            let select = select
                .as_deref()
                .map(parse_select)
                .unwrap_or(ChromosomeSelect::All);
            let ldsc_log = ldsc_log.map(PathBuf::from);
            let config = LdscConfig {
                traits,
                sample_prev,
                population_prev,
                ld,
                wld,
                trait_names,
                sep_weights,
                chr,
                n_blocks,
                ldsc_log,
                stand,
                select,
                chisq_max,
                output,
            };
            ldsc(&config)?;
        }
        Command::SLdsc {
            traits,
            sample_prev,
            population_prev,
            ld,
            wld,
            frq,
            trait_names,
            n_blocks,
            ldsc_log,
            exclude_cont,
            output,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let sample_prev = sample_prev.map(parse_optional_f64_list);
            let population_prev = population_prev.map(parse_optional_f64_list);
            let config = SLdscConfig {
                traits,
                sample_prev,
                population_prev,
                ld,
                wld,
                frq,
                trait_names,
                n_blocks,
                ldsc_log,
                exclude_cont,
            };
            let out = s_ldsc(&config)?;
            let output = output.unwrap_or_else(|| PathBuf::from("s_ldsc.json"));
            write_s_ldsc_json(&out, &output)?;
        }
        Command::Enrich {
            s_covstruc,
            model,
            model_file,
            params,
            fix,
            std_lv,
            rm_flank,
            tau,
            base,
            toler,
            fixparam,
            output_prefix,
        } => {
            let s_covstruc = read_s_ldsc_json(&s_covstruc)?;
            let model =
                resolve_model_string(model, model_file, "enrich requires --model or --model-file")?;
            let params = split_string_list(params);
            let fixparam = fixparam.map(split_string_list);
            let config = EnrichConfig {
                model,
                params,
                fix,
                std_lv,
                rm_flank,
                tau,
                base,
                toler,
                fixparam,
            };
            let out = enrich(&s_covstruc, &config)?;
            let prefix = output_prefix.unwrap_or_else(|| "enrich".to_string());
            if out.results.len() == 1 && !base {
                let path = PathBuf::from(format!("{prefix}.tsv"));
                write_dataframe(&out.results[0], &path)?;
            } else {
                for (idx, df) in out.results.iter().enumerate() {
                    let path = PathBuf::from(format!("{prefix}_param{}.tsv", idx + 1));
                    write_dataframe(df, &path)?;
                }
            }
            if let Some(base_df) = out.base_results {
                let path = PathBuf::from(format!("{prefix}_base.tsv"));
                write_dataframe(&base_df, &path)?;
            }
        }
        Command::ReadFusion {
            files,
            trait_names,
            binary,
            n,
            perm,
            output,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let binary = binary.map(|v| split_bool_list(&v));
            let n = n.map(split_f64_list);
            let config = ReadFusionConfig {
                files,
                trait_names,
                binary,
                n,
                perm,
            };
            let out = read_fusion(&config)?;
            let output = output.unwrap_or_else(|| PathBuf::from("twas_sumstats.tsv"));
            write_dataframe(&out.df, &output)?;
        }
        Command::CommonfactorGwas {
            covstruc,
            snps,
            estimation,
            cores,
            toler,
            snp_se,
            parallel,
            gc,
            mpi,
            twas,
            smooth_check,
            output,
        } => {
            let covstruc = read_ldsc_json(&covstruc)?;
            let df = read_table(&snps)?;
            let snps = SumstatsTable { df };
            let config = CommonFactorGwasConfig {
                estimation: parse_estimation(&estimation)?,
                cores,
                toler: toler.or(Some(f64::EPSILON)),
                snp_se,
                parallel,
                gc: parse_gc(&gc)?,
                mpi,
                twas,
                smooth_check,
            };
            let out = commonfactor_gwas(&covstruc, &snps, &config)?;
            let df = commonfactor_gwas_output_table(&out)?;
            let output = output.unwrap_or_else(|| PathBuf::from("commonfactorGWAS.tsv"));
            write_dataframe(&df, &output)?;
        }
        Command::UserGwas {
            covstruc,
            snps,
            model,
            model_file,
            estimation,
            printwarn,
            sub,
            cores,
            toler,
            snp_se,
            parallel,
            gc,
            mpi,
            smooth_check,
            twas,
            std_lv,
            fix_measurement,
            q_snp,
            output,
        } => {
            let covstruc = read_ldsc_json(&covstruc)?;
            let df = read_table(&snps)?;
            let snps = SumstatsTable { df };
            let model = resolve_model_string(
                model,
                model_file,
                "user-gwas requires --model or --model-file",
            )?;
            let sub = sub.map(split_string_list).and_then(|vals| {
                let cleaned: Vec<String> = vals
                    .into_iter()
                    .map(|s| s.replace(' ', ""))
                    .filter(|s| !s.is_empty())
                    .collect();
                if cleaned.is_empty()
                    || (cleaned.len() == 1 && cleaned[0].eq_ignore_ascii_case("FALSE"))
                {
                    None
                } else {
                    Some(cleaned)
                }
            });
            let config = UserGwasConfig {
                estimation: parse_estimation(&estimation)?,
                model,
                printwarn,
                sub: sub.clone(),
                cores,
                toler: toler.or(Some(f64::EPSILON)),
                snp_se,
                parallel,
                gc: parse_gc(&gc)?,
                mpi,
                smooth_check,
                twas,
                std_lv,
                fix_measurement,
                q_snp,
            };
            let out = user_gwas(&covstruc, &snps, &config)?;
            let df = user_gwas_output_table(&out)?;
            let output = output.unwrap_or_else(|| PathBuf::from("userGWAS.tsv"));
            if let Some(subs) = sub.as_ref() {
                for (idx, key) in subs.iter().enumerate() {
                    let filtered = filter_user_gwas_sub(&df, key)?;
                    let path = output_with_sub_suffix(&output, idx + 1);
                    write_dataframe(&filtered, &path)?;
                }
            } else {
                write_dataframe(&df, &output)?;
            }
        }
        Command::Commonfactor {
            covstruc,
            estimation,
            output_prefix,
        } => {
            let covstruc = read_ldsc_json(&covstruc)?;
            let estimation = parse_estimation(&estimation)?;
            let out = commonfactor(&covstruc, estimation)?;
            if out.smoothed_s {
                tracing::warn!(
                    "S matrix was smoothed prior to model estimation; max abs diff={}",
                    out.ld_sdiff
                );
            }
            if out.smoothed_v {
                tracing::warn!(
                    "V matrix was smoothed prior to model estimation; max abs diff={}",
                    out.ld_sdiff2
                );
            }
            let (modelfit, results) = commonfactor_output_tables(&out)?;
            let prefix = output_prefix.unwrap_or_else(|| "commonfactor".to_string());
            write_dataframe(&modelfit, &PathBuf::from(format!("{prefix}_modelfit.tsv")))?;
            write_dataframe(&results, &PathBuf::from(format!("{prefix}_results.tsv")))?;
        }
        Command::Usermodel {
            covstruc,
            model,
            model_file,
            estimation,
            std_lv,
            cfi_calc,
            output_prefix,
        } => {
            let covstruc = read_ldsc_json(&covstruc)?;
            let model = resolve_model_string(
                model,
                model_file,
                "usermodel requires --model or --model-file",
            )?;
            let estimation = parse_estimation(&estimation)?;
            let out = usermodel(&covstruc, &model, estimation, std_lv, cfi_calc)?;
            if out.smoothed_s {
                tracing::warn!(
                    "S matrix was smoothed prior to model estimation; max abs diff={}",
                    out.ld_sdiff
                );
            }
            if out.smoothed_v {
                tracing::warn!(
                    "V matrix was smoothed prior to model estimation; max abs diff={}",
                    out.ld_sdiff2
                );
            }
            let (modelfit, results) = usermodel_output_tables(&out)?;
            let prefix = output_prefix.unwrap_or_else(|| "usermodel".to_string());
            write_dataframe(&modelfit, &PathBuf::from(format!("{prefix}_modelfit.tsv")))?;
            write_dataframe(&results, &PathBuf::from(format!("{prefix}_results.tsv")))?;
        }
        Command::Hdl {
            traits,
            sample_prev,
            population_prev,
            ld_path,
            n_ref,
            trait_names,
            method,
            output,
        } => {
            let trait_names = trait_names.map(split_string_list);
            let sample_prev = sample_prev.map(parse_optional_f64_list).unwrap_or_default();
            let population_prev = population_prev
                .map(parse_optional_f64_list)
                .unwrap_or_default();
            let method = parse_hdl_method(&method)?;
            let config = HdlConfig {
                traits,
                sample_prev,
                population_prev,
                ld_path,
                n_ref,
                trait_names,
                method,
            };
            let out = hdl(&config)?;
            let output = output.unwrap_or_else(|| PathBuf::from("hdl_ldsc.json"));
            write_ldsc_json(&out, &output)?;
        }
        Command::PaLdsc {
            s,
            v,
            r,
            p,
            diag,
            fa,
            nfactors,
            save_plots,
            output_prefix,
        } => {
            let s = read_matrix_file(&s)?;
            let v = read_matrix_file(&v)?;
            let out = pa_ldsc(&s, &v, r, p, diag, fa, nfactors)?;
            let prefix = output_prefix
                .clone()
                .unwrap_or_else(|| "pa_ldsc".to_string());
            write_pa_ldsc(&out, &prefix)?;
            if save_plots {
                write_pa_ldsc_plots(&out, &s, output_prefix.as_deref())?;
            }
        }
        Command::IndexS {
            s,
            include_diag,
            output,
        } => {
            let s = read_matrix_file(&s)?;
            let idx = index_s_from_matrix(&s, include_diag);
            let output = output.unwrap_or_else(|| PathBuf::from("indexS.tsv"));
            write_matrix(&idx, &output)?;
        }
        Command::SubSv {
            s,
            v,
            index,
            ty,
            output_prefix,
        } => {
            let s = read_matrix_file(&s)?;
            let v = read_matrix_file(&v)?;
            let index_vals = split_usize_list(&index);
            let ty = parse_subsv_type(&ty)?;
            match ty {
                SubSvType::S => {}
                SubSvType::SStand => {
                    tracing::warn!(
                        "sub-sv type S_Stand selected: ensure provided S/V are S_Stand/V_Stand"
                    );
                }
                SubSvType::R => {
                    tracing::warn!(
                        "sub-sv type R selected: R/V_R are not implemented in Rust; using provided matrices as-is"
                    );
                }
            }
            let out = sub_sv_from_matrices(&s, &v, &index_vals)?;
            let prefix = output_prefix.unwrap_or_else(|| "subsv".to_string());
            write_vector(&out.sub_s, &PathBuf::from(format!("{prefix}_subS.tsv")))?;
            write_matrix(&out.sub_v, &PathBuf::from(format!("{prefix}_subV.tsv")))?;
        }
        Command::SummaryGls {
            y,
            v,
            predictors,
            intercept,
            output,
        } => {
            let y = read_vector_file(&y)?;
            let v = read_matrix_file(&v)?;
            let predictors = read_matrix_file(&predictors)?;
            let out = summary_gls(&y, &v, &predictors, intercept)?;
            let output = output.unwrap_or_else(|| PathBuf::from("summary_gls.tsv"));
            write_summary_gls(&out, &output)?;
        }
        Command::SummaryGlsBands {
            y,
            v,
            predictors,
            controlvars,
            intervals,
            intercept,
            quad,
            bands,
            no_bands,
            band_size,
            save_plots,
            xlab,
            ylab,
            title,
            xcoords,
            ycoords,
            output_prefix,
        } => {
            let y = read_vector_file(&y)?;
            let v = read_matrix_file(&v)?;
            let predictors = read_matrix_file(&predictors)?;
            let controlvars = if let Some(path) = controlvars {
                Some(read_matrix_file(&path)?)
            } else {
                None
            };
            let bands = if no_bands { false } else { bands };
            let out = summary_gls_bands(
                &y,
                &v,
                &predictors,
                intervals,
                controlvars.as_deref(),
                intercept,
                quad,
                bands,
                band_size,
            )?;
            let prefix = output_prefix
                .clone()
                .unwrap_or_else(|| "summary_gls_bands".to_string());
            write_summary_gls(&out.gls, &PathBuf::from(format!("{prefix}_gls.tsv")))?;
            if let (Some(pred), Some(upper), Some(lower)) = (
                out.band_predictors.as_ref(),
                out.band_upper.as_ref(),
                out.band_lower.as_ref(),
            ) {
                write_band_table(
                    pred,
                    upper,
                    lower,
                    &PathBuf::from(format!("{prefix}_bands.tsv")),
                )?;
            }
            if save_plots {
                let x_range = parse_axis_range(xcoords, [-2.0, 2.0]);
                let y_range = parse_axis_range(ycoords, [0.0, 1.0]);
                write_summary_gls_bands_plot(
                    &y,
                    &predictors,
                    &out,
                    intervals,
                    quad,
                    intercept,
                    bands,
                    &xlab,
                    &ylab,
                    &title,
                    x_range,
                    y_range,
                    output_prefix.as_deref(),
                )?;
            }
        }
        Command::LocalSrmd {
            unconstrained,
            constrained,
            lhsvar,
            rhsvar,
            output_prefix,
        } => {
            let unconstrained = read_vector_file(&unconstrained)?;
            let constrained = read_vector_file(&constrained)?;
            let lhsvar = read_matrix_file(&lhsvar)?;
            let rhsvar = read_matrix_file(&rhsvar)?;
            let out = local_srmd(&unconstrained, &constrained, &lhsvar, &rhsvar)?;
            let prefix = output_prefix.unwrap_or_else(|| "localsrmd".to_string());
            write_vector(
                &out.localdiff,
                &PathBuf::from(format!("{prefix}_localdiff.tsv")),
            )?;
            write_scalar(out.value, &PathBuf::from(format!("{prefix}_value.txt")))?;
        }
        Command::SimLdsc {
            covmat,
            n,
            ld,
            r_pheno,
            intercepts,
            n_overlap,
            r,
            seed,
            gzip_output,
            parallel,
            cores,
        } => {
            let covmat = CovInput::Matrix(read_matrix_file(&covmat)?);
            let n = parse_n_input(&n)?;
            let r_pheno = r_pheno.map(parse_matrix_or_scalar).transpose()?;
            let intercepts = intercepts.map(parse_vec_or_scalar).transpose()?;
            let config = SimLdscConfig {
                covmat,
                n,
                seed,
                ld,
                r_pheno,
                intercepts,
                n_overlap,
                r,
                gzip_output,
                parallel,
                cores,
            };
            sim_ldsc(&config)?;
        }
        Command::QTrait {
            ldsc,
            indicators,
            traits,
            mresid,
            mresid_threshold,
            lsrmr,
            lsrmr_threshold,
            save_plots,
            stdout,
            output,
        } => {
            let ldsc = read_ldsc_json(&ldsc)?;
            let indicators = split_string_list(indicators);
            let traits = split_string_list(traits);
            let config = QTraitConfig {
                indicators,
                traits,
                mresid,
                mresid_threshold,
                lsrmr,
                lsrmr_threshold,
                save_plots,
                stdout,
            };
            let df = qtrait(&ldsc, &config)?;
            let output = output.unwrap_or_else(|| PathBuf::from("qtrait.tsv"));
            write_dataframe(&df, &output)?;
        }
    }

    Ok(())
}

fn split_string_list(input: String) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn split_f64_list(input: String) -> Vec<f64> {
    split_string_list(input)
        .into_iter()
        .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
        .collect()
}

fn parse_axis_range(input: String, default: [f64; 2]) -> [f64; 2] {
    let vals = split_f64_list(input);
    if vals.len() >= 2 && vals[0].is_finite() && vals[1].is_finite() {
        [vals[0], vals[1]]
    } else {
        default
    }
}

fn split_bool_list(input: &str) -> Vec<bool> {
    split_string_list(input.to_string())
        .into_iter()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "T"))
        .collect()
}

fn parse_optional_f64_list(input: String) -> Vec<Option<f64>> {
    split_string_list(input)
        .into_iter()
        .map(|s| {
            if s.trim().is_empty() || s.eq_ignore_ascii_case("NA") {
                None
            } else {
                s.parse::<f64>().ok()
            }
        })
        .collect()
}

fn parse_select(input: &str) -> ChromosomeSelect {
    let upper = input.trim().to_ascii_uppercase();
    if upper == "ODD" {
        return ChromosomeSelect::Odd;
    }
    if upper == "EVEN" {
        return ChromosomeSelect::Even;
    }
    let list = split_string_list(input.to_string())
        .into_iter()
        .filter_map(|s| s.parse::<u8>().ok())
        .collect::<Vec<_>>();
    if list.is_empty() {
        ChromosomeSelect::All
    } else {
        ChromosomeSelect::List(list)
    }
}

// read_matrix_file/read_vector_file/write_matrix/write_vector/write_scalar/write_dataframe live in io.rs

fn write_summary_gls(out: &SummaryGlsOutput, path: &PathBuf) -> anyhow::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "betas\tpvals\tSE\tZ")?;
    let n = out.betas.len();
    for i in 0..n {
        let b = out.betas.get(i).copied().unwrap_or(f64::NAN);
        let p = out.pvals.get(i).copied().unwrap_or(f64::NAN);
        let se = out.se.get(i).copied().unwrap_or(f64::NAN);
        let z = out.z.get(i).copied().unwrap_or(f64::NAN);
        writeln!(file, "{b}\t{p}\t{se}\t{z}")?;
    }
    Ok(())
}

fn write_band_table(
    predictors: &[f64],
    upper: &[f64],
    lower: &[f64],
    path: &PathBuf,
) -> anyhow::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "Predictors\tUpper\tLower")?;
    for (i, x) in predictors.iter().enumerate() {
        let u = upper.get(i).copied().unwrap_or(f64::NAN);
        let l = lower.get(i).copied().unwrap_or(f64::NAN);
        writeln!(file, "{x}\t{u}\t{l}")?;
    }
    Ok(())
}

fn write_pa_ldsc(out: &genomicsem::post_ldsc::PaLdscOutput, prefix: &str) -> anyhow::Result<()> {
    let path = PathBuf::from(format!("{prefix}.tsv"));
    let mut file = File::create(&path)?;
    writeln!(file, "Observed\tParallel")?;
    let n = out.observed.len().max(out.parallel.len());
    for i in 0..n {
        let o = out.observed.get(i).copied().unwrap_or(f64::NAN);
        let p = out.parallel.get(i).copied().unwrap_or(f64::NAN);
        writeln!(file, "{o}\t{p}")?;
    }
    if let (Some(obs), Some(par)) = (&out.observed_diag, &out.parallel_diag) {
        let path = PathBuf::from(format!("{prefix}_diag.tsv"));
        let mut file = File::create(&path)?;
        writeln!(file, "Observed\tParallel")?;
        let n = obs.len().max(par.len());
        for i in 0..n {
            let o = obs.get(i).copied().unwrap_or(f64::NAN);
            let p = par.get(i).copied().unwrap_or(f64::NAN);
            writeln!(file, "{o}\t{p}")?;
        }
    }
    if let (Some(obs), Some(par)) = (&out.observed_fa, &out.parallel_fa) {
        let path = PathBuf::from(format!("{prefix}_fa.tsv"));
        let mut file = File::create(&path)?;
        writeln!(file, "Observed\tParallel")?;
        let n = obs.len().max(par.len());
        for i in 0..n {
            let o = obs.get(i).copied().unwrap_or(f64::NAN);
            let p = par.get(i).copied().unwrap_or(f64::NAN);
            writeln!(file, "{o}\t{p}")?;
        }
    }
    if let (Some(obs), Some(par)) = (&out.observed_fa_diag, &out.parallel_fa_diag) {
        let path = PathBuf::from(format!("{prefix}_fa_diag.tsv"));
        let mut file = File::create(&path)?;
        writeln!(file, "Observed\tParallel")?;
        let n = obs.len().max(par.len());
        for i in 0..n {
            let o = obs.get(i).copied().unwrap_or(f64::NAN);
            let p = par.get(i).copied().unwrap_or(f64::NAN);
            writeln!(file, "{o}\t{p}")?;
        }
    }
    Ok(())
}

fn output_with_sub_suffix(base: &Path, idx: usize) -> PathBuf {
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("userGWAS");
    let ext = base.extension().and_then(|s| s.to_str());
    let file_name = if let Some(ext) = ext {
        format!("{stem}_sub{idx}.{ext}")
    } else {
        format!("{stem}_sub{idx}")
    };
    base.with_file_name(file_name)
}

fn resolve_model_string(
    model: Option<String>,
    model_file: Option<PathBuf>,
    missing_msg: &str,
) -> anyhow::Result<String> {
    if let Some(path) = model_file {
        Ok(std::fs::read_to_string(path).context("read model file")?)
    } else if let Some(model) = model {
        Ok(model)
    } else {
        Err(anyhow::anyhow!(missing_msg.to_string()))
    }
}

fn filter_user_gwas_sub(df: &DataFrame, key: &str) -> anyhow::Result<DataFrame> {
    let lhs = df.column("lhs")?.as_materialized_series();
    let op = df.column("op")?.as_materialized_series();
    let rhs = df.column("rhs")?.as_materialized_series();
    let mut mask = Vec::with_capacity(df.height());
    for idx in 0..df.height() {
        let l = lhs.get(idx).ok();
        let o = op.get(idx).ok();
        let r = rhs.get(idx).ok();
        let keep = match (l, o, r) {
            (Some(AnyValue::String(l)), Some(AnyValue::String(o)), Some(AnyValue::String(r))) => {
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::StringOwned(l)),
                Some(AnyValue::StringOwned(o)),
                Some(AnyValue::StringOwned(r)),
            ) => {
                let l = l.as_str();
                let o = o.as_str();
                let r = r.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::String(l)),
                Some(AnyValue::StringOwned(o)),
                Some(AnyValue::StringOwned(r)),
            ) => {
                let o = o.as_str();
                let r = r.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::StringOwned(l)),
                Some(AnyValue::String(o)),
                Some(AnyValue::StringOwned(r)),
            ) => {
                let l = l.as_str();
                let r = r.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::StringOwned(l)),
                Some(AnyValue::StringOwned(o)),
                Some(AnyValue::String(r)),
            ) => {
                let l = l.as_str();
                let o = o.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::String(l)),
                Some(AnyValue::String(o)),
                Some(AnyValue::StringOwned(r)),
            ) => {
                let r = r.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::String(l)),
                Some(AnyValue::StringOwned(o)),
                Some(AnyValue::String(r)),
            ) => {
                let o = o.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            (
                Some(AnyValue::StringOwned(l)),
                Some(AnyValue::String(o)),
                Some(AnyValue::String(r)),
            ) => {
                let l = l.as_str();
                let mut joined = String::with_capacity(l.len() + o.len() + r.len());
                joined.push_str(l);
                joined.push_str(o);
                joined.push_str(r);
                joined.is_empty() || joined == key
            }
            _ => false,
        };
        mask.push(keep);
    }
    let mask = BooleanChunked::from_iter(mask);
    Ok(df.filter(&mask)?)
}

fn min_max(values: &[f64]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values {
        if v.is_finite() {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}

fn min_max_many(series: &[&[f64]]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for values in series {
        if let Some((smin, smax)) = min_max(values) {
            min = min.min(smin);
            max = max.max(smax);
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}

fn add_vertical_line(plot: &mut Plot, x: f64, y_min: f64, y_max: f64) {
    let line = Line::default()
        .color(NamedColor::DarkGray)
        .dash(DashType::Dash);
    let trace = Scatter::new(vec![x, x], vec![y_min, y_max])
        .mode(Mode::Lines)
        .line(line)
        .show_legend(false);
    plot.add_trace(trace);
}

fn add_horizontal_line(plot: &mut Plot, y: f64, x_min: f64, x_max: f64) {
    let line = Line::default()
        .color(NamedColor::DarkGray)
        .dash(DashType::Dash);
    let trace = Scatter::new(vec![x_min, x_max], vec![y, y])
        .mode(Mode::Lines)
        .line(line)
        .show_legend(false);
    plot.add_trace(trace);
}

#[allow(clippy::too_many_arguments)]
fn write_summary_gls_bands_plot(
    y: &[f64],
    predictors: &[Vec<f64>],
    out: &SummaryGlsBandsOutput,
    intervals: usize,
    quad: bool,
    intercept: bool,
    bands: bool,
    xlab: &str,
    ylab: &str,
    title: &str,
    xcoords: [f64; 2],
    ycoords: [f64; 2],
    prefix: Option<&str>,
) -> anyhow::Result<()> {
    ensure_plots_dir()?;
    if predictors.is_empty() || y.is_empty() {
        return Ok(());
    }
    let x_vals: Vec<f64> = predictors.iter().map(|row| row[0]).collect();
    let y_vals: Vec<f64> = y.to_vec();

    let min_x = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_x - min_x;
    let x_seq = if let Some(pred) = out.band_predictors.as_ref() {
        pred.clone()
    } else {
        (0..intervals)
            .map(|i| min_x + (i as f64) * range / (intervals as f64))
            .collect()
    };

    let line_x = if xcoords[0].is_finite() && xcoords[1].is_finite() && xcoords[0] != xcoords[1] {
        let n_line = 200usize;
        (0..n_line)
            .map(|i| xcoords[0] + (i as f64) * (xcoords[1] - xcoords[0]) / ((n_line - 1) as f64))
            .collect::<Vec<_>>()
    } else {
        x_seq.clone()
    };

    let base_idx = if intercept { 1 } else { 0 };
    let b0 = if intercept {
        out.gls.betas.first().copied().unwrap_or(f64::NAN)
    } else {
        0.0
    };
    let b1 = out.gls.betas.get(base_idx).copied().unwrap_or(f64::NAN);
    let b2 = if quad {
        out.gls.betas.get(base_idx + 1).copied().unwrap_or(f64::NAN)
    } else {
        0.0
    };
    let fit_line = line_x
        .iter()
        .map(|x| {
            if quad {
                b0 + b1 * x + b2 * x * x
            } else {
                b0 + b1 * x
            }
        })
        .collect::<Vec<_>>();

    let mut plot = Plot::new();

    if bands && let (Some(upper), Some(lower)) = (&out.band_upper, &out.band_lower) {
        let band_color = Rgb::new(0x7A, 0x90, 0x83);
        let lower_trace = Scatter::new(x_seq.clone(), lower.clone())
            .mode(Mode::Lines)
            .line(
                Line::default()
                    .color(band_color)
                    .dash(DashType::Dot)
                    .width(1.0),
            )
            .show_legend(false);
        let upper_trace = Scatter::new(x_seq.clone(), upper.clone())
            .mode(Mode::Lines)
            .line(
                Line::default()
                    .color(band_color)
                    .dash(DashType::Dot)
                    .width(1.0),
            )
            .fill(Fill::ToNextY)
            .fill_color(Rgba::new(0x7A, 0x90, 0x83, 0.1))
            .show_legend(false);
        plot.add_trace(lower_trace);
        plot.add_trace(upper_trace);
    }

    let fit_trace = Scatter::new(line_x, fit_line)
        .mode(Mode::Lines)
        .line(
            Line::default()
                .color(Rgb::new(0x7A, 0x90, 0x83))
                .dash(DashType::Dash)
                .width(2.0),
        )
        .name("Fit");
    plot.add_trace(fit_trace);

    let marker_base = Marker::new()
        .color(NamedColor::Black)
        .size(8)
        .symbol(MarkerSymbol::Circle);
    let marker_inner = Marker::new()
        .color(Rgb::new(0xDF, 0xD0, 0xB7))
        .size(5)
        .symbol(MarkerSymbol::Circle);
    let points_base = Scatter::new(x_vals.clone(), y_vals.clone())
        .mode(Mode::Markers)
        .marker(marker_base)
        .name("Observed");
    let points_inner = Scatter::new(x_vals, y_vals)
        .mode(Mode::Markers)
        .marker(marker_inner)
        .show_legend(false);
    plot.add_trace(points_base);
    plot.add_trace(points_inner);

    let mut x_axis = Axis::new().title(xlab.to_string());
    let mut y_axis = Axis::new().title(ylab.to_string());
    if xcoords[0].is_finite() && xcoords[1].is_finite() {
        x_axis = x_axis.range(vec![xcoords[0], xcoords[1]]);
    }
    if ycoords[0].is_finite() && ycoords[1].is_finite() {
        y_axis = y_axis.range(vec![ycoords[0], ycoords[1]]);
    }
    let mut layout = Layout::new().x_axis(x_axis).y_axis(y_axis);
    if !title.is_empty() {
        layout = layout.title(title.to_string());
    }
    plot.set_layout(layout);

    let path = plot_path(prefix, "summaryGLSbands");
    plot.write_html(path);
    Ok(())
}

fn write_pa_ldsc_plots(
    out: &genomicsem::post_ldsc::PaLdscOutput,
    s: &Matrix,
    prefix: Option<&str>,
) -> anyhow::Result<()> {
    ensure_plots_dir()?;

    let k = out.observed.len();
    if k == 0 {
        return Ok(());
    }
    let x_vals: Vec<f64> = (1..=k).map(|v| v as f64).collect();
    let diag_is_one = s
        .iter()
        .enumerate()
        .all(|(i, row)| row.get(i).copied().unwrap_or(f64::NAN) == 1.0);

    let obs_minus = out
        .observed
        .iter()
        .zip(&out.parallel)
        .map(|(o, p)| o - p)
        .collect::<Vec<_>>();
    let nfact_obs = obs_minus
        .iter()
        .position(|v| v.is_finite() && *v < 0.0)
        .map(|idx| idx as f64);
    let nfact = out.n_factors.map(|v| v as f64);

    let mut plot = Plot::new();
    let observed_trace = Scatter::new(x_vals.clone(), out.observed.clone())
        .mode(Mode::LinesMarkers)
        .marker(
            Marker::new()
                .symbol(MarkerSymbol::TriangleUp)
                .color(NamedColor::Black)
                .size(8),
        )
        .line(Line::default().color(NamedColor::Black))
        .name("Observed");
    let parallel_trace = Scatter::new(x_vals.clone(), out.parallel.clone())
        .mode(Mode::LinesMarkers)
        .marker(
            Marker::new()
                .symbol(MarkerSymbol::TriangleUpOpen)
                .color(NamedColor::DarkGray)
                .size(8),
        )
        .line(Line::default().color(NamedColor::DarkGray))
        .name("Simulated");
    plot.add_trace(observed_trace);
    plot.add_trace(parallel_trace);

    let (y_min, y_max) = min_max_many(&[&out.observed, &out.parallel]).unwrap_or((0.0, 1.0));
    let pad = (y_max - y_min).abs() * 0.05;
    let y_min = y_min - pad;
    let y_max = y_max + pad;
    if diag_is_one {
        add_horizontal_line(&mut plot, 1.0, 1.0, k as f64);
    } else if let Some(vline) = nfact {
        add_vertical_line(&mut plot, vline, y_min, y_max);
    }

    let x_axis = Axis::new()
        .title("Component Number")
        .tick_values(x_vals.clone());
    let y_axis = Axis::new().title("Eigenvalue");
    plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
    plot.write_html(plot_path(prefix, "PA_LDSC"));

    let mut plot = Plot::new();
    let obs_trace = Scatter::new(x_vals.clone(), obs_minus.clone())
        .mode(Mode::LinesMarkers)
        .marker(
            Marker::new()
                .symbol(MarkerSymbol::TriangleUp)
                .color(NamedColor::Black)
                .size(8),
        )
        .line(Line::default().color(NamedColor::Black))
        .name("Observed minus simulated data");
    plot.add_trace(obs_trace);
    let (y_min, y_max) = min_max(&obs_minus).unwrap_or((-1.0, 1.0));
    let pad = (y_max - y_min).abs() * 0.05;
    add_horizontal_line(&mut plot, 0.0, 1.0, k as f64);
    if let Some(vline) = nfact_obs
        && vline > 0.0
    {
        add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
    }
    let x_axis = Axis::new()
        .title("Component Number")
        .tick_values(x_vals.clone());
    let y_axis = Axis::new().title("Difference in Eigenvalues");
    plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
    plot.write_html(plot_path(prefix, "PA_LDSC_obs_minus"));

    if let (Some(obs_diag), Some(par_diag)) = (&out.observed_diag, &out.parallel_diag) {
        let k = obs_diag.len();
        let x_vals: Vec<f64> = (1..=k).map(|v| v as f64).collect();
        let obs_minus = obs_diag
            .iter()
            .zip(par_diag)
            .map(|(o, p)| o - p)
            .collect::<Vec<_>>();
        let nfact = out.n_factors_diag.map(|v| v as f64);
        let nfact_obs = obs_minus
            .iter()
            .position(|v| v.is_finite() && *v < 0.0)
            .map(|idx| idx as f64);

        let mut plot = Plot::new();
        let observed_trace = Scatter::new(x_vals.clone(), obs_diag.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed");
        let parallel_trace = Scatter::new(x_vals.clone(), par_diag.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUpOpen)
                    .color(NamedColor::DarkGray)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::DarkGray))
            .name("Simulated");
        plot.add_trace(observed_trace);
        plot.add_trace(parallel_trace);
        let (y_min, y_max) = min_max_many(&[obs_diag, par_diag]).unwrap_or((0.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 1.0, 1.0, k as f64);
        if let Some(vline) = nfact {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new()
            .title("Component Number")
            .tick_values(x_vals.clone());
        let y_axis = Axis::new().title("Diagonalized Eigenvalue");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "Diagonalized_PA_LDSC"));

        let mut plot = Plot::new();
        let obs_trace = Scatter::new(x_vals.clone(), obs_minus.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed minus simulated data");
        plot.add_trace(obs_trace);
        let (y_min, y_max) = min_max(&obs_minus).unwrap_or((-1.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 0.0, 1.0, k as f64);
        if let Some(vline) = nfact_obs
            && vline > 0.0
        {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new()
            .title("Component Number")
            .tick_values(x_vals.clone());
        let y_axis = Axis::new().title("Difference in diagonalized Eigenvalues");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "Diagonalized_PA_LDSC_obs_minus"));
    }

    if let (Some(obs_fa), Some(par_fa)) = (&out.observed_fa, &out.parallel_fa) {
        let k = obs_fa.len();
        let x_vals: Vec<f64> = (1..=k).map(|v| v as f64).collect();
        let obs_minus = obs_fa
            .iter()
            .zip(par_fa)
            .map(|(o, p)| o - p)
            .collect::<Vec<_>>();
        let nfact = out.n_factors_fa.map(|v| v as f64);
        let nfact_obs = obs_minus
            .iter()
            .position(|v| v.is_finite() && *v < 0.0)
            .map(|idx| idx as f64);

        let mut plot = Plot::new();
        let observed_trace = Scatter::new(x_vals.clone(), obs_fa.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed");
        let parallel_trace = Scatter::new(x_vals.clone(), par_fa.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUpOpen)
                    .color(NamedColor::DarkGray)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::DarkGray))
            .name("Simulated");
        plot.add_trace(observed_trace);
        plot.add_trace(parallel_trace);
        let (y_min, y_max) = min_max_many(&[obs_fa, par_fa]).unwrap_or((0.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 1.0, 1.0, k as f64);
        if let Some(vline) = nfact {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new()
            .title("Factor Number")
            .tick_values(x_vals.clone());
        let y_axis = Axis::new().title("Eigenvalue from FA Solution");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "FA_PA_LDSC"));

        let mut plot = Plot::new();
        let obs_trace = Scatter::new(x_vals.clone(), obs_minus.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed minus simulated data");
        plot.add_trace(obs_trace);
        let (y_min, y_max) = min_max(&obs_minus).unwrap_or((-1.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 0.0, 1.0, k as f64);
        if let Some(vline) = nfact_obs
            && vline > 0.0
        {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new()
            .title("Factor Number")
            .tick_values(x_vals.clone());
        let y_axis = Axis::new().title("Difference in FA Eigenvalues");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "FA_PA_LDSC_obs_minus"));
    }

    if let (Some(obs_fa), Some(par_fa)) = (&out.observed_fa_diag, &out.parallel_fa_diag) {
        let k = obs_fa.len();
        let x_vals: Vec<f64> = (1..=k).map(|v| v as f64).collect();
        let obs_minus = obs_fa
            .iter()
            .zip(par_fa)
            .map(|(o, p)| o - p)
            .collect::<Vec<_>>();
        let nfact = out.n_factors_fa_diag.map(|v| v as f64);
        let nfact_obs = obs_minus
            .iter()
            .position(|v| v.is_finite() && *v < 0.0)
            .map(|idx| idx as f64);

        let mut plot = Plot::new();
        let observed_trace = Scatter::new(x_vals.clone(), obs_fa.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed");
        let parallel_trace = Scatter::new(x_vals.clone(), par_fa.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUpOpen)
                    .color(NamedColor::DarkGray)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::DarkGray))
            .name("Simulated");
        plot.add_trace(observed_trace);
        plot.add_trace(parallel_trace);
        let (y_min, y_max) = min_max_many(&[obs_fa, par_fa]).unwrap_or((0.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 1.0, 1.0, k as f64);
        if let Some(vline) = nfact {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new()
            .title("Factor Number")
            .tick_values(x_vals.clone());
        let y_axis = Axis::new().title("Eigenvalue from diagonalized FA Solution");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "FA_Diagonalized_PA_LDSC"));

        let mut plot = Plot::new();
        let obs_trace = Scatter::new(x_vals.clone(), obs_minus.clone())
            .mode(Mode::LinesMarkers)
            .marker(
                Marker::new()
                    .symbol(MarkerSymbol::TriangleUp)
                    .color(NamedColor::Black)
                    .size(8),
            )
            .line(Line::default().color(NamedColor::Black))
            .name("Observed minus simulated data");
        plot.add_trace(obs_trace);
        let (y_min, y_max) = min_max(&obs_minus).unwrap_or((-1.0, 1.0));
        let pad = (y_max - y_min).abs() * 0.05;
        add_horizontal_line(&mut plot, 0.0, 1.0, k as f64);
        if let Some(vline) = nfact_obs
            && vline > 0.0
        {
            add_vertical_line(&mut plot, vline, y_min - pad, y_max + pad);
        }
        let x_axis = Axis::new().title("Factor Number").tick_values(x_vals);
        let y_axis = Axis::new().title("Difference in diagonalized FA Eigenvalues");
        plot.set_layout(Layout::new().x_axis(x_axis).y_axis(y_axis));
        plot.write_html(plot_path(prefix, "FA_Diagonalized_PA_LDSC_obs_minus"));
    }

    Ok(())
}

// write_ldsc_json lives in io.rs

fn parse_hdl_method(input: &str) -> anyhow::Result<HdlMethod> {
    let lower = input.to_ascii_lowercase();
    match lower.as_str() {
        "piecewise" => Ok(HdlMethod::Piecewise),
        "jackknife" => Ok(HdlMethod::Jackknife),
        _ => Err(anyhow::anyhow!("Unknown HDL method: {input}")),
    }
}

fn parse_estimation(input: &str) -> anyhow::Result<Estimation> {
    let upper = input.to_ascii_uppercase();
    match upper.as_str() {
        "DWLS" => Ok(Estimation::Dwls),
        "ML" => Ok(Estimation::Ml),
        _ => Err(anyhow::anyhow!("Unknown estimation method: {input}")),
    }
}

fn parse_gc(input: &str) -> anyhow::Result<GenomicControl> {
    let lower = input.to_ascii_lowercase();
    match lower.as_str() {
        "standard" | "stand" => Ok(GenomicControl::Standard),
        "conserv" | "conservative" => Ok(GenomicControl::Conserv),
        "none" => Ok(GenomicControl::None),
        _ => Err(anyhow::anyhow!("Unknown GC method: {input}")),
    }
}

fn parse_subsv_type(input: &str) -> anyhow::Result<SubSvType> {
    let upper = input.to_ascii_uppercase();
    match upper.as_str() {
        "S" => Ok(SubSvType::S),
        "S_STAND" | "SSTAND" => Ok(SubSvType::SStand),
        "R" => Ok(SubSvType::R),
        _ => Err(anyhow::anyhow!("Unknown subSV type: {input}")),
    }
}

fn split_usize_list(input: &str) -> Vec<usize> {
    split_string_list(input.to_string())
        .into_iter()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect()
}

fn parse_n_input(input: &str) -> anyhow::Result<NInput> {
    if let Ok(val) = input.parse::<f64>() {
        return Ok(NInput::Scalar(val));
    }
    let path = PathBuf::from(input);
    let matrix = read_matrix_file(&path)?;
    Ok(NInput::Matrix(matrix))
}

fn parse_matrix_or_scalar(input: String) -> anyhow::Result<MatrixOrScalar> {
    if let Ok(val) = input.parse::<f64>() {
        return Ok(MatrixOrScalar::Scalar(val));
    }
    let path = PathBuf::from(input);
    let matrix = read_matrix_file(&path)?;
    Ok(MatrixOrScalar::Matrix(matrix))
}

fn parse_vec_or_scalar(input: String) -> anyhow::Result<VecOrScalar> {
    if let Ok(val) = input.parse::<f64>() {
        return Ok(VecOrScalar::Scalar(val));
    }
    let path = PathBuf::from(input);
    let vec = read_vector_file(&path)?;
    Ok(VecOrScalar::Vec(vec))
}
