# SEMMS Benchmark Suite
#
# Runs fitSEMMS() on all bundled example datasets and records:
#   - Wall-clock time (user + system + elapsed via system.time)
#   - Mixture model parameters (mu, s2e, s2r)
#   - Number of selected variables and their indices
#   - For simulated datasets: TP, FP, FN, precision, recall
#   - AIC of the downstream GLM fit
#
# Results are saved as an RDS file (timestamped) under inst/benchmarks/results/
# and also appended to a running CSV log for easy tracking across versions.
#
# Usage (from the package root):
#   Rscript inst/benchmarks/run_benchmarks.R
# Or from an R session:
#   source("inst/benchmarks/run_benchmarks.R")

library(SEMMS)

results_dir <- file.path("inst", "benchmarks", "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

timestamp   <- format(Sys.time(), "%Y%m%d_%H%M%S")
semms_ver   <- as.character(packageVersion("SEMMS"))
r_ver       <- paste(R.version$major, R.version$minor, sep = ".")

cat("=== SEMMS Benchmark Suite ===\n")
cat(sprintf("Date       : %s\n", timestamp))
cat(sprintf("SEMMS ver  : %s\n", semms_ver))
cat(sprintf("R version  : %s\n\n", r_ver))

# ---------------------------------------------------------------------------
# Helper: classification metrics for simulated datasets
# true_idx  : integer vector of 1-based column indices that are truly non-null
# found_idx : integer vector of 1-based indices selected by SEMMS
# ---------------------------------------------------------------------------
classif_metrics <- function(true_idx, found_idx) {
  tp <- length(intersect(found_idx, true_idx))
  fp <- length(setdiff(found_idx, true_idx))
  fn <- length(setdiff(true_idx, found_idx))
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA
  recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA
  f1        <- if (!is.na(precision) && !is.na(recall) &&
                     (precision + recall) > 0)
                 2 * precision * recall / (precision + recall) else NA
  list(tp = tp, fp = fp, fn = fn, precision = precision,
       recall = recall, f1 = f1)
}

# ---------------------------------------------------------------------------
# Helper: run one benchmark scenario and return a named list of results
# ---------------------------------------------------------------------------
run_one <- function(label, N, P, true_idx = NULL,
                    load_data_fn, fit_params) {
  cat(sprintf("--- %s  (N=%d, P=%d) ---\n", label, N, P))

  dat <- load_data_fn()

  t_fit <- system.time({
    fit <- do.call(fitSEMMS, c(list(dat = dat), fit_params))
  })

  nn_found <- sort(fit$gam.out$nn)   # 1-based column indices

  t_glm <- system.time({
    glm_fit <- if (length(nn_found) > 0)
      runLinearModel(dat, nn_found, fit_params$distribution)
    else
      NULL
  })

  aic_val <- if (!is.null(glm_fit)) glm_fit$aic else NA

  cm <- if (!is.null(true_idx))
    classif_metrics(true_idx, nn_found)
  else
    list(tp = NA, fp = NA, fn = NA, precision = NA, recall = NA, f1 = NA)

  cat(sprintf("  elapsed (fit): %.2f s | elapsed (glm): %.2f s\n",
              t_fit["elapsed"], t_glm["elapsed"]))
  cat(sprintf("  selected: %d vars | AIC: %.1f\n",
              length(nn_found), aic_val))
  if (!is.null(true_idx))
    cat(sprintf("  TP=%d  FP=%d  FN=%d  precision=%.3f  recall=%.3f  F1=%.3f\n",
                cm$tp, cm$fp, cm$fn,
                ifelse(is.na(cm$precision), 0, cm$precision),
                ifelse(is.na(cm$recall),    0, cm$recall),
                ifelse(is.na(cm$f1),        0, cm$f1)))
  cat("\n")

  list(
    label        = label,
    N            = N,
    P            = P,
    distribution = fit_params$distribution,
    mincor       = fit_params$mincor,
    nn_init      = fit_params$nn,
    minchange    = fit_params$minchange,
    initEF       = isTRUE(fit_params$initWithEdgeFinder),
    time_fit_user    = t_fit["user.self"],
    time_fit_sys     = t_fit["sys.self"],
    time_fit_elapsed = t_fit["elapsed"],
    time_glm_elapsed = t_glm["elapsed"],
    n_selected   = length(nn_found),
    selected_idx = list(nn_found),
    mu           = fit$gam.out$mu,
    s2e          = fit$gam.out$s2e,
    s2r          = fit$gam.out$s2r,
    aic          = aic_val,
    tp           = cm$tp,
    fp           = cm$fp,
    fn           = cm$fn,
    precision    = cm$precision,
    recall       = cm$recall,
    f1           = cm$f1,
    semms_ver    = semms_ver,
    r_ver        = r_ver,
    timestamp    = timestamp
  )
}

# ===========================================================================
# 1. Ozone  (N=330, P=54 after 2nd-order interactions, continuous response)
# ===========================================================================
res_ozone <- run_one(
  label    = "Ozone",
  N        = 330,
  P        = 54,
  true_idx = NULL,          # real dataset – no ground truth
  load_data_fn = function() {
    fn <- system.file("extdata", "ozone.txt", package = "SEMMS", mustWork = TRUE)
    readInputFile(fn, ycol = 2, skip = 19, Zcols = 3:11,
                  addIntercept = TRUE, logTransform = 2, twoWay = TRUE)
  },
  fit_params = list(
    distribution       = "N",
    mincor             = 0.75,
    nn                 = 20,
    minchange          = 1,
    initWithEdgeFinder = FALSE,
    rnd                = FALSE,
    verbose            = FALSE
  )
)

# ===========================================================================
# 2. NKI70  (N=144, P=72, Poisson/survival response)
# ===========================================================================
res_nki70 <- run_one(
  label    = "NKI70",
  N        = 144,
  P        = 72,
  true_idx = NULL,          # real dataset – no ground truth
  load_data_fn = function() {
    fn <- system.file("extdata", "NKI70_t1.RData", package = "SEMMS", mustWork = TRUE)
    readInputFile(file = fn, ycol = 1, Zcols = 2:73)
  },
  fit_params = list(
    distribution       = "P",
    mincor             = 0.8,
    nn                 = 6,
    minchange          = 1,
    initWithEdgeFinder = TRUE,
    rnd                = FALSE,
    verbose            = FALSE
  )
)

# ===========================================================================
# 3. AR1SIM  (N=100, P=1000, continuous response)
#    True predictors: Z1–Z20  (first hub of 20 variables, 1-based indices 1:20)
# ===========================================================================
res_ar1sim <- run_one(
  label    = "AR1SIM",
  N        = 100,
  P        = 1000,
  true_idx = 1:20,
  load_data_fn = function() {
    fn <- system.file("extdata", "AR1SIM.RData", package = "SEMMS", mustWork = TRUE)
    readInputFile(fn, ycol = 1, Zcols = 2:1001)
  },
  fit_params = list(
    distribution       = "N",
    mincor             = 0.8,
    nn                 = 15,
    minchange          = 1,
    initWithEdgeFinder = TRUE,
    rnd                = FALSE,
    verbose            = FALSE
  )
)

# ===========================================================================
# 4. SimBin  (N=100, P=1000, binary response)
#    True predictors: Z1–Z5 and Z101–Z105  (1-based indices)
# ===========================================================================
res_simbin <- run_one(
  label    = "SimBin",
  N        = 100,
  P        = 1000,
  true_idx = c(1:5, 101:105),
  load_data_fn = function() {
    fn <- system.file("extdata", "SimBin.RData", package = "SEMMS", mustWork = TRUE)
    readInputFile(fn, ycol = 1, Zcols = 2:1001)
  },
  fit_params = list(
    distribution       = "B",
    mincor             = 0.7,
    nn                 = 5,
    minchange          = 1,
    initWithEdgeFinder = TRUE,
    rnd                = FALSE,
    verbose            = FALSE
  )
)

# ===========================================================================
# 5. Simulated dataset: N=500, P=1000, 6 true predictors, Normal response
#
# Design:
#   - True predictors (indices 1–6) drawn from AR(1) MVN with rho=0.5
#   - Remaining 994 predictors drawn i.i.d. N(0,1)
#   - beta_true = c(1.0, -1.0, 0.8, -0.8, 0.6, -0.6)  =>  R² ≈ 0.80
#   - Noise sigma_e = 1.0
#   - Fixed seed for reproducibility; dataset saved to inst/extdata/ so
#     every benchmark run uses the identical realisation.
# ===========================================================================

sim_file <- file.path("inst", "extdata", "sim_N500_P1000_k6.RData")

if (!file.exists(sim_file)) {
  set.seed(20260314)
  N_sim  <- 500L
  P_sim  <- 1000L
  n_true <- 6L

  # All predictors i.i.d. N(0,1) — avoids the marginal-correlation cancellation
  # that occurs with AR(1) + alternating-sign betas, where correlated neighbours
  # partially cancel in their joint projection onto Y.
  Z_raw <- matrix(rnorm(N_sim * P_sim), N_sim, P_sim)
  colnames(Z_raw) <- paste0("V", seq_len(P_sim))

  # Mixed-sign betas, all >= 0.6 in magnitude  =>  R² ≈ 0.75
  beta_true <- c(0.8, -0.7, 0.6, -0.8, 0.7, -0.6)
  Y_raw <- Z_raw[, seq_len(n_true)] %*% beta_true + rnorm(N_sim, 0, 1)

  sim_N500_P1000_k6 <- data.frame(Y = Y_raw, Z_raw)
  save(sim_N500_P1000_k6, file = sim_file)
  cat(sprintf("Simulated dataset generated and saved to: %s\n\n", sim_file))
}

# Verify R² before handing off to SEMMS
{
  tmp_env <- new.env()
  load(sim_file, envir = tmp_env)
  tmp_dat <- tmp_env$sim_N500_P1000_k6
  lm_oracle <- lm(Y ~ ., data = tmp_dat[, 1:7])   # Y + 6 true predictors
  cat(sprintf("Oracle R² (true 6 predictors only): %.3f\n\n",
              summary(lm_oracle)$r.squared))
  rm(tmp_env, tmp_dat, lm_oracle)
}

res_sim500 <- run_one(
  label    = "Sim_N500_P1000_k6",
  N        = 500L,
  P        = 1000L,
  true_idx = 1:6,
  load_data_fn = function() {
    readInputFile(sim_file, ycol = 1, Zcols = 2:1001)
  },
  fit_params = list(
    distribution       = "N",
    mincor             = 0.7,
    nn                 = 6,
    minchange          = 1,
    initWithEdgeFinder = TRUE,
    rnd                = FALSE,
    verbose            = FALSE
  )
)

# ===========================================================================
# Save results
# ===========================================================================
all_results <- list(
  ozone      = res_ozone,
  nki70      = res_nki70,
  ar1sim     = res_ar1sim,
  simbin     = res_simbin,
  sim_N500   = res_sim500
)

rds_path <- file.path(results_dir, sprintf("benchmark_%s.rds", timestamp))
saveRDS(all_results, rds_path)
cat(sprintf("Full results saved to: %s\n\n", rds_path))

# Append a flat summary row to the running CSV log
csv_path <- file.path(results_dir, "benchmark_log.csv")

flat_row <- do.call(rbind, lapply(all_results, function(r) {
  data.frame(
    timestamp        = r$timestamp,
    semms_ver        = r$semms_ver,
    r_ver            = r$r_ver,
    label            = r$label,
    N                = r$N,
    P                = r$P,
    distribution     = r$distribution,
    mincor           = r$mincor,
    nn_init          = r$nn_init,
    minchange        = r$minchange,
    initEF           = r$initEF,
    time_fit_elapsed = r$time_fit_elapsed,
    time_glm_elapsed = r$time_glm_elapsed,
    n_selected       = r$n_selected,
    mu               = r$mu,
    s2e              = r$s2e,
    s2r              = r$s2r,
    aic              = r$aic,
    tp               = r$tp,
    fp               = r$fp,
    fn               = r$fn,
    precision        = r$precision,
    recall           = r$recall,
    f1               = r$f1,
    stringsAsFactors = FALSE
  )
}))

write.table(flat_row,
            file      = csv_path,
            append    = file.exists(csv_path),
            sep       = ",",
            row.names = FALSE,
            col.names = !file.exists(csv_path),
            quote     = TRUE)

cat(sprintf("CSV log updated : %s\n", csv_path))

# Print summary table to console
cat("\n=== Summary ===\n")
print(flat_row[, c("label", "N", "P", "time_fit_elapsed",
                   "n_selected", "aic", "tp", "fp", "fn",
                   "precision", "recall", "f1")],
      row.names = FALSE, digits = 3)
