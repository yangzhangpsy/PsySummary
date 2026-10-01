#!/usr/bin/env python3
"""Compare rtdists/nlminb and PsySummary fits on rtdists-generated data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.cognitiveModels import fit_cognitive_model
from app.cognitiveModelSpec import (
    LBA_MODEL,
    RATCLIFF_MODEL,
    RDM_MODEL,
    make_model_specification,
)


R_VALIDATION_PROGRAM = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Expected arguments: output_directory sample_size random_seed")
}

output_directory <- normalizePath(args[[1]], mustWork = FALSE)
sample_size <- as.integer(args[[2]])
random_seed <- as.integer(args[[3]])
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(rtdists))
rtdists_version <- as.character(packageVersion("rtdists"))

safe_nll <- function(densities) {
  if (any(!is.finite(densities)) || any(densities <= 0)) return(1e100)
  -sum(log(densities))
}

fit_rows <- list()
append_fit <- function(model, fit, all_parameters, true_parameters, sample_n) {
  fit_rows[[length(fit_rows) + 1]] <<- data.frame(
    model = model,
    parameter = names(all_parameters),
    true_value = as.numeric(true_parameters[names(all_parameters)]),
    estimate = as.numeric(all_parameters),
    nll = as.numeric(fit$objective),
    converged = fit$convergence == 0,
    convergence_code = as.integer(fit$convergence),
    message = if (is.null(fit$message)) "" else as.character(fit$message),
    sample_size = sample_n,
    optimizer = "nlminb",
    rtdists_version = rtdists_version,
    random_seed = random_seed,
    stringsAsFactors = FALSE
  )
}

# Ratcliff diffusion model: a, v, and t0 are free; variability and scale are fixed.
set.seed(random_seed)
ddm_true <- c(a = 1.4, v = 0.8, t0 = 0.25, z = 0.7, d = 0, sz = 0, sv = 0, st0 = 0, s = 1)
ddm_data <- rdiffusion(
  sample_size, a = ddm_true[["a"]], v = ddm_true[["v"]], t0 = ddm_true[["t0"]],
  z = ddm_true[["z"]], d = ddm_true[["d"]], sz = ddm_true[["sz"]],
  sv = ddm_true[["sv"]], st0 = ddm_true[["st0"]], s = ddm_true[["s"]]
)
ddm_data$response <- as.character(ddm_data$response)
write.csv(ddm_data, file.path(output_directory, "ratcliff_diffusion_data.csv"), row.names = FALSE)
ddm_nll <- function(par) {
  names(par) <- c("a", "v", "t0")
  if (par[["a"]] <= ddm_true[["z"]] || par[["t0"]] < 0) return(1e100)
  safe_nll(ddiffusion(
    ddm_data, a = par[["a"]], v = par[["v"]], t0 = par[["t0"]],
    z = ddm_true[["z"]], d = 0, sz = 0, sv = 0, st0 = 0, s = 1,
    precision = 7
  ))
}
ddm_upper_t0 <- min(ddm_data$rt) * 0.999
ddm_fit <- nlminb(
  start = c(a = 1.0, v = 0.3, t0 = 0.15), objective = ddm_nll,
  lower = c(a = 0.71, v = -5, t0 = 0.001),
  upper = c(a = 3, v = 5, t0 = ddm_upper_t0),
  control = list(iter.max = 1000, eval.max = 3000, rel.tol = 1e-10)
)
ddm_parameters <- ddm_true
ddm_parameters[names(ddm_fit$par)] <- ddm_fit$par
append_fit("Ratcliff Diffusion Model", ddm_fit, ddm_parameters, ddm_true, nrow(ddm_data))

# Linear ballistic accumulator: drift-rate scales are fixed for identifiability.
set.seed(random_seed + 1L)
lba_true <- c(A = 0.5, b = 1.2, t0 = 0.2, st0 = 0,
              `mean_v[1]` = 2.4, `sd_v[1]` = 1, `mean_v[2]` = 1.2, `sd_v[2]` = 1)
lba_data <- suppressMessages(rLBA(
  sample_size, A = lba_true[["A"]], b = lba_true[["b"]], t0 = lba_true[["t0"]],
  mean_v = c(lba_true[["mean_v[1]"]], lba_true[["mean_v[2]"]]),
  sd_v = c(1, 1), st0 = 0, silent = TRUE
))
lba_data$response <- as.integer(as.character(lba_data$response))
write.csv(lba_data, file.path(output_directory, "linear_ballistic_accumulator_data.csv"), row.names = FALSE)
lba_nll <- function(par) {
  names(par) <- c("A", "b", "t0", "mean_v[1]", "mean_v[2]")
  if (par[["A"]] < 0 || par[["b"]] <= par[["A"]] || par[["t0"]] < 0) return(1e100)
  safe_nll(dLBA(
    lba_data, A = par[["A"]], b = par[["b"]], t0 = par[["t0"]],
    mean_v = c(par[["mean_v[1]"]], par[["mean_v[2]"]]),
    sd_v = c(1, 1), st0 = 0, silent = TRUE
  ))
}
lba_upper_t0 <- min(lba_data$rt) * 0.999
lba_fit <- nlminb(
  start = c(A = 0.3, b = 1.0, t0 = 0.15, `mean_v[1]` = 2.0, `mean_v[2]` = 1.5),
  objective = lba_nll,
  lower = c(A = 0, b = 0.05, t0 = 0.001, `mean_v[1]` = 0.01, `mean_v[2]` = 0.01),
  upper = c(A = 2, b = 3, t0 = lba_upper_t0, `mean_v[1]` = 6, `mean_v[2]` = 6),
  control = list(iter.max = 1000, eval.max = 3000, rel.tol = 1e-10)
)
lba_parameters <- lba_true
lba_parameters[names(lba_fit$par)] <- lba_fit$par
append_fit("Linear Ballistic Accumulator", lba_fit, lba_parameters, lba_true, nrow(lba_data))

# Racing diffusion model: diffusion scale is fixed for identifiability.
set.seed(random_seed + 2L)
rdm_true <- c(A = 0.4, b = 1.1, t0 = 0.2, st0 = 0, s = 1, `v[1]` = 2.2, `v[2]` = 1.2)
rdm_data <- suppressMessages(rRDM(
  sample_size, A = rdm_true[["A"]], b = rdm_true[["b"]], t0 = rdm_true[["t0"]],
  v = c(rdm_true[["v[1]"]], rdm_true[["v[2]"]]), s = 1, st0 = 0, silent = TRUE
))
rdm_data$response <- as.integer(as.character(rdm_data$response))
write.csv(rdm_data, file.path(output_directory, "racing_diffusion_data.csv"), row.names = FALSE)
rdm_nll <- function(par) {
  names(par) <- c("A", "b", "t0", "v[1]", "v[2]")
  if (par[["A"]] < 0 || par[["b"]] <= par[["A"]] || par[["t0"]] < 0 ||
      any(par[c("v[1]", "v[2]")] < 0)) return(1e100)
  safe_nll(dRDM(
    rdm_data, A = par[["A"]], b = par[["b"]], t0 = par[["t0"]],
    v = c(par[["v[1]"]], par[["v[2]"]]), s = 1, st0 = 0, silent = TRUE
  ))
}
rdm_upper_t0 <- min(rdm_data$rt) * 0.999
rdm_fit <- nlminb(
  start = c(A = 0.3, b = 1.0, t0 = 0.15, `v[1]` = 2.0, `v[2]` = 1.5),
  objective = rdm_nll,
  lower = c(A = 0, b = 0.05, t0 = 0.001, `v[1]` = 0.001, `v[2]` = 0.001),
  upper = c(A = 2, b = 3, t0 = rdm_upper_t0, `v[1]` = 6, `v[2]` = 6),
  control = list(iter.max = 1000, eval.max = 3000, rel.tol = 1e-10)
)
rdm_parameters <- rdm_true
rdm_parameters[names(rdm_fit$par)] <- rdm_fit$par
append_fit("Racing Diffusion Model", rdm_fit, rdm_parameters, rdm_true, nrow(rdm_data))

write.csv(do.call(rbind, fit_rows), file.path(output_directory, "rtdists_fits.csv"), row.names = FALSE)
cat(sprintf("rtdists %s validation data and fits written to %s\n", rtdists_version, output_directory))
'''


MODEL_CONFIGURATIONS = {
    RATCLIFF_MODEL: {
        'file': 'ratcliff_diffusion_data.csv',
        'responses': ['lower', 'upper'],
        'free': {
            'a': (1.0, 0.71, 3.0),
            'v': (0.3, -5.0, 5.0),
            't0': (0.15, 0.001, None),
        },
        'fixed': {'z': 0.7, 'd': 0.0, 'sz': 0.0, 'sv': 0.0, 'st0': 0.0, 's': 1.0},
    },
    LBA_MODEL: {
        'file': 'linear_ballistic_accumulator_data.csv',
        'responses': [1, 2],
        'free': {
            'A': (0.3, 0.0, 2.0),
            'b': (1.0, 0.05, 3.0),
            't0': (0.15, 0.001, None),
            'mean_v[1]': (2.0, 0.01, 6.0),
            'mean_v[2]': (1.5, 0.01, 6.0),
        },
        'fixed': {'st0': 0.0, 'sd_v[1]': 1.0, 'sd_v[2]': 1.0},
    },
    RDM_MODEL: {
        'file': 'racing_diffusion_data.csv',
        'responses': [1, 2],
        'free': {
            'A': (0.3, 0.0, 2.0),
            'b': (1.0, 0.05, 3.0),
            't0': (0.15, 0.001, None),
            'v[1]': (2.0, 0.001, 6.0),
            'v[2]': (1.5, 0.001, 6.0),
        },
        'fixed': {'st0': 0.0, 's': 1.0},
    },
}


def _parse_arguments():
    """Parse command-line options for the validation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=500,
                        help='Number of rtdists-generated trials per model (default: 500).')
    parser.add_argument('--seed', type=int, default=8102,
                        help='Base simulation and PsySummary optimizer seed (default: 8102).')
    parser.add_argument('--rscript', default='Rscript', help='Path to the Rscript executable.')
    parser.add_argument('--r-library', type=Path,
                        help='Optional R library containing the rtdists package.')
    parser.add_argument('--output-dir', type=Path,
                        default=SCRIPT_DIR / 'rtdists_fit_validation',
                        help='Directory for datasets and comparison reports.')
    parser.add_argument('--parameter-tolerance', type=float, default=1e-2,
                        help='Maximum free-parameter difference considered aligned.')
    parser.add_argument('--nll-tolerance', type=float, default=1e-2,
                        help='Maximum NLL difference considered aligned.')
    arguments = parser.parse_args()
    if arguments.sample_size < 20:
        parser.error('--sample-size must be at least 20.')
    return arguments


def _run_rtdists(arguments, output_directory):
    """Generate datasets and rtdists/nlminb fits through Rscript."""
    executable = shutil.which(arguments.rscript) if not Path(arguments.rscript).is_file() else arguments.rscript
    if not executable:
        raise RuntimeError(f"Rscript executable not found: {arguments.rscript}")
    environment = os.environ.copy()
    if arguments.r_library:
        environment['R_LIBS_USER'] = str(arguments.r_library.resolve())
    with tempfile.NamedTemporaryFile('w', suffix='.R', encoding='utf-8', delete=False) as handle:
        handle.write(R_VALIDATION_PROGRAM)
        temporary_program = Path(handle.name)
    try:
        completed = subprocess.run(
            [str(executable), str(temporary_program), str(output_directory),
             str(arguments.sample_size), str(arguments.seed)],
            check=True, capture_output=True, text=True, env=environment,
        )
    except subprocess.CalledProcessError as error:
        details = '\n'.join(part for part in (error.stdout, error.stderr) if part)
        raise RuntimeError(f'rtdists validation failed in R:\n{details}') from error
    finally:
        temporary_program.unlink(missing_ok=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def _make_psysummary_specification(model, dataframe, optimizer_seed):
    """Build the PsySummary specification matching the R validation fit."""
    configuration = MODEL_CONFIGURATIONS[model]
    specification = make_model_specification(
        model, 'rt', 'response', configuration['responses'],
        minimum_rt=float(dataframe['rt'].min()),
        accuracy_variable='accuracy' if model == RATCLIFF_MODEL else None,
        boundary_coding='response',
    )
    by_name = {parameter['name']: parameter for parameter in specification['parameters']}
    for parameter in specification['parameters']:
        parameter['mode'] = 'fixed'
    for name, value in configuration['fixed'].items():
        by_name[name]['value'] = value
    upper_t0 = float(dataframe['rt'].min()) * 0.999
    for name, (value, lower, upper) in configuration['free'].items():
        by_name[name].update(
            mode='free', value=value, lower=lower,
            upper=upper_t0 if upper is None else upper,
        )
    specification['optimizer'].update(
        method='SLSQP', starts=1, seed=optimizer_seed, max_iterations=1000)
    return specification


def _fit_with_psysummary(output_directory, optimizer_seed):
    """Fit all generated datasets through PsySummary's production fitter."""
    rows = []
    for model, configuration in MODEL_CONFIGURATIONS.items():
        dataframe = pd.read_csv(output_directory / configuration['file'])
        if model == RATCLIFF_MODEL:
            dataframe['accuracy'] = (dataframe['response'].astype(str) == 'upper').astype(int)
        specification = _make_psysummary_specification(model, dataframe, optimizer_seed)
        fit = fit_cognitive_model(dataframe, specification)
        for parameter, estimate, mode in zip(
                fit['parameter_names'], fit['parameters'], fit['parameter_modes']):
            rows.append({
                'model': model,
                'parameter': parameter,
                'estimate': estimate,
                'mode': mode,
                'nll': -fit['log_likelihood'],
                'converged': fit['converged'],
                'message': fit['message'],
                'sample_size': fit['n_valid'],
                'optimizer': fit['optimizer_method'],
                'random_seed': fit['random_seed'],
            })
    result = pd.DataFrame(rows)
    result.to_csv(output_directory / 'psysummary_fits.csv', index=False)
    return result


def _build_comparison(rtdists, psysummary, parameter_tolerance, nll_tolerance):
    """Combine R and PsySummary fit rows and classify numerical agreement."""
    r_columns = {
        'estimate': 'rtdists_estimate', 'nll': 'rtdists_nll',
        'converged': 'rtdists_converged', 'message': 'rtdists_message',
        'optimizer': 'rtdists_optimizer',
    }
    p_columns = {
        'estimate': 'psysummary_estimate', 'nll': 'psysummary_nll',
        'converged': 'psysummary_converged', 'message': 'psysummary_message',
        'optimizer': 'psysummary_optimizer',
    }
    left = rtdists.rename(columns=r_columns)
    right = psysummary.rename(columns=p_columns)
    comparison = left.merge(
        right[['model', 'parameter', 'mode', *p_columns.values()]],
        on=['model', 'parameter'], how='outer', validate='one_to_one',
    )
    comparison['estimate_difference'] = (
        comparison['psysummary_estimate'] - comparison['rtdists_estimate'])
    comparison['absolute_estimate_difference'] = comparison['estimate_difference'].abs()
    comparison['nll_difference'] = comparison['psysummary_nll'] - comparison['rtdists_nll']
    comparison['absolute_nll_difference'] = comparison['nll_difference'].abs()
    comparison['within_tolerance'] = (
        comparison['rtdists_converged'].astype(bool)
        & comparison['psysummary_converged'].astype(bool)
        & (comparison['absolute_nll_difference'] <= nll_tolerance)
        & (comparison['absolute_estimate_difference'] <= parameter_tolerance)
    )
    return comparison


def _markdown_report(comparison, arguments):
    """Render a readable validation report from the long comparison table."""
    rtdists_version = comparison['rtdists_version'].dropna().astype(str).iloc[0]
    lines = [
        '# rtdists and PsySummary fit validation', '',
        f'- Simulation: `rtdists {rtdists_version}`',
        '- R fit: `rtdists` likelihood functions with `nlminb`',
        '- PsySummary fit: production `fit_cognitive_model()` with constrained `SLSQP`',
        f'- Trials per model: `{arguments.sample_size}`',
        f'- Base random seed: `{arguments.seed}`',
        f'- Free-parameter tolerance: `{arguments.parameter_tolerance:g}`',
        f'- NLL tolerance: `{arguments.nll_tolerance:g}`', '',
    ]
    for model in MODEL_CONFIGURATIONS:
        subset = comparison.loc[comparison['model'] == model].copy()
        status = 'PASS' if bool(subset['within_tolerance'].all()) else 'REVIEW'
        lines.extend([
            f'## {model}', '',
            f'Overall status: **{status}**', '',
            f"- rtdists/nlminb converged: `{bool(subset['rtdists_converged'].iloc[0])}`",
            f"- PsySummary/SLSQP converged: `{bool(subset['psysummary_converged'].iloc[0])}`",
            f"- rtdists NLL: `{subset['rtdists_nll'].iloc[0]:.10f}`",
            f"- PsySummary NLL: `{subset['psysummary_nll'].iloc[0]:.10f}`",
            f"- Absolute NLL difference: `{subset['absolute_nll_difference'].iloc[0]:.6g}`",
            '',
            '| Parameter | Mode | True | rtdists | PsySummary | Absolute difference |',
            '| --- | --- | ---: | ---: | ---: | ---: |',
        ])
        for row in subset.itertuples():
            lines.append(
                f'| {row.parameter} | {row.mode} | {row.true_value:.6g} | '
                f'{row.rtdists_estimate:.10g} | {row.psysummary_estimate:.10g} | '
                f'{row.absolute_estimate_difference:.6g} |')
        lines.append('')
    lines.extend([
        '## Interpretation', '',
        'Random samples are identical across the two fits because both consume the CSV files generated by R. '
        'The optimizers are intentionally different, so iteration paths and messages need not match. '
        'The comparison evaluates convergence, final negative log-likelihood, and free-parameter estimates.', '',
    ])
    return '\n'.join(lines)


def main():
    """Run simulation, both fitting implementations, and report generation."""
    arguments = _parse_arguments()
    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _run_rtdists(arguments, output_directory)
    rtdists = pd.read_csv(output_directory / 'rtdists_fits.csv')
    psysummary = _fit_with_psysummary(output_directory, arguments.seed)
    comparison = _build_comparison(
        rtdists, psysummary, arguments.parameter_tolerance, arguments.nll_tolerance)
    comparison_path = output_directory / 'fit_comparison.csv'
    report_path = output_directory / 'fit_comparison.md'
    comparison.to_csv(comparison_path, index=False)
    report_path.write_text(_markdown_report(comparison, arguments), encoding='utf-8')
    overall_status = 'PASS' if bool(comparison['within_tolerance'].all()) else 'REVIEW'
    print(f'Validation status: {overall_status}')
    print(f'Comparison CSV: {comparison_path}')
    print(f'Markdown report: {report_path}')
    return 0 if overall_status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
