# Generate the rtdists 0.12-0 values used by the Python cognitive-model tests.
# Run with: Rscript generate_rtdists_cognitive_reference.R

library(rtdists)

if (as.character(packageVersion("rtdists")) != "0.12.0") {
  stop("This reference must be generated with rtdists 0.12-0.")
}

times <- c(0.3, 0.5, 0.8, 1.2)
probability <- 0.25

print_case <- function(name, density, distribution, quantile, parameters, responses) {
  cat("\n", name, "\n", sep="")
  for (response in responses) {
    cat("response:", response, "\n")
    cat("pdf:", format(do.call(density, c(list(
      rt=times, response=response, silent=TRUE), parameters)), digits=16), "\n")
    cat("cdf@0.8:", format(do.call(distribution, c(list(
      rt=0.8, response=response, silent=TRUE), parameters)), digits=16), "\n")
    cat("q@0.25:", format(do.call(quantile, c(list(
      p=probability, response=response, silent=TRUE, interval=c(0, 10),
      scale_p=TRUE, scale_max=10), parameters)), digits=16), "\n")
  }
}

ddm_baseline <- list(a=1, v=1, t0=0.2, z=0.4, d=0, sz=0, sv=0, st0=0, s=1)
ddm_variability <- list(
  a=1.3, v=-0.7, t0=0.25, z=0.6, d=0.04, sz=0.2, sv=0.5, st0=0.1, s=0.8)
ddm_cases <- list(baseline=ddm_baseline, variability=ddm_variability)
for (case_name in names(ddm_cases)) {
  entry <- ddm_cases[[case_name]]
  cdf_precision <- if (case_name == "baseline") 7 else 5
  cat("\nDDM ", case_name, "\n", sep="")
  for (response in c("lower", "upper")) {
    cat("response:", response, "\n")
    cat("pdf:", format(do.call(ddiffusion, c(list(
      rt=times, response=response, precision=7), entry)), digits=16), "\n")
    cat("cdf@0.8:", format(do.call(pdiffusion, c(list(
      rt=0.8, response=response, precision=cdf_precision), entry)), digits=16), "\n")
    cat("q@0.25:", format(do.call(qdiffusion, c(list(
      p=probability, response=response, precision=3, interval=c(0, 10),
      scale_p=TRUE, scale_max=10), entry)), digits=16), "\n")
  }
}

print_case(
  "LBA", dLBA, pLBA, qLBA,
  list(A=0.5, b=1, t0=0.2, mean_v=c(2, 1), sd_v=c(1, 1), st0=0), 1:2)
print_case(
  "RDM", dRDM, pRDM, qRDM,
  list(A=0.5, b=1, t0=0.2, v=c(2, 1), s=1, st0=0), 1:2)
