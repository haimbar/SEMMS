# Generate test dataset for the SEMMS mixed-effects extension
#
# Design
# ------
#   N = 200 observations: 20 subjects x 10 time points each
#   K = 100 predictors (Z), all i.i.d. N(0,1)
#   True signal: columns 1–5 only
#     beta_true = c(1.5, -1.2, 1.0, -0.9, 0.8)
#
#   Random effects (subject-level):
#     Random intercept  b0_i ~ N(0, sigma_b0^2),  sigma_b0 = 1.5
#     Random slope      b1_i ~ N(0, sigma_b1^2),  sigma_b1 = 0.5  (on scaled time)
#     Residual noise            eps  ~ N(0, sigma_e^2),   sigma_e  = 1.0
#
#   Data-generating model:
#     Y_ij = Z_ij[1:5] %*% beta_true  +  b0_i  +  b1_i * time_scaled_j  +  eps_ij
#
# Column layout of the saved data frame:
#   col 1 : Y          (response)
#   col 2 : subject    (integer 1–20, grouping factor)
#   col 3 : time       (integer 1–10, random slope variable)
#   col 4–103 : V1–V100 (predictors; these become Zcols in readInputFile)
#
# Intended use with fitSEMMSmixed():
#   dat <- readInputFile(fn, ycol = 1, Zcols = 4:103)
#   # subject is col 2, time col 3 — passed as group_col / random_slope_col
#   fit <- fitSEMMSmixed(dat, ...,
#                        group_col        = "subject",
#                        random_intercept = TRUE,
#                        random_slope_col = "time")

out_file <- file.path("inst", "extdata", "sim_mixed_N200_P100_k5.RData")

set.seed(20260314)

N_subj   <- 20L
N_time   <- 10L
N        <- N_subj * N_time   # 200
K        <- 100L
K_true   <- 5L

sigma_b0 <- 1.5   # random intercept SD
sigma_b1 <- 0.5   # random slope SD
sigma_e  <- 1.0   # residual SD

# ── Subject / time indices ───────────────────────────────────────────────────
subject <- rep(seq_len(N_subj), each  = N_time)
time    <- rep(seq_len(N_time), times = N_subj)

# Scale time to mean 0, SD 1 so slope variance is on a comparable scale
time_scaled <- as.numeric(scale(time))

# ── Random effects ───────────────────────────────────────────────────────────
b0 <- rnorm(N_subj, mean = 0, sd = sigma_b0)   # random intercepts
b1 <- rnorm(N_subj, mean = 0, sd = sigma_b1)   # random slopes

re_contribution <- b0[subject] + b1[subject] * time_scaled

# ── Predictors ───────────────────────────────────────────────────────────────
Z <- matrix(rnorm(N * K), nrow = N, ncol = K)
colnames(Z) <- paste0("V", seq_len(K))

# ── Fixed effects ────────────────────────────────────────────────────────────
beta_true <- c(1.5, -1.2, 1.0, -0.9, 0.8)
fixed_contribution <- Z[, seq_len(K_true)] %*% beta_true

# ── Response ─────────────────────────────────────────────────────────────────
Y <- as.numeric(fixed_contribution) + re_contribution + rnorm(N, 0, sigma_e)

# ── Assemble data frame ───────────────────────────────────────────────────────
sim_mixed_N200_P100_k5 <- data.frame(
  Y       = Y,
  subject = subject,
  time    = time,
  Z
)

save(sim_mixed_N200_P100_k5, file = out_file)
cat(sprintf("Dataset saved to: %s\n", out_file))

# ── Sanity checks ─────────────────────────────────────────────────────────────

# 1. Oracle R² from fixed effects only (ignoring RE)
lm_oracle_fixed <- lm(Y ~ Z[, 1:K_true])
cat(sprintf("Oracle R² (fixed effects only, ignoring RE): %.3f\n",
            summary(lm_oracle_fixed)$r.squared))

# 2. Oracle lmer R² (fixed + RE) — needs lme4
if (requireNamespace("lme4", quietly = TRUE)) {
  library(lme4)
  df_check <- sim_mixed_N200_P100_k5
  true_vars <- paste(paste0("V", 1:K_true), collapse = " + ")
  f_oracle  <- as.formula(
    sprintf("Y ~ %s + (1 + time_scaled | subject)", true_vars)
  )
  df_check$time_scaled <- time_scaled
  lmer_oracle <- lmer(f_oracle, data = df_check, REML = FALSE)
  # marginal R² (fixed effects) and conditional R² (fixed + RE)
  if (requireNamespace("performance", quietly = TRUE)) {
    r2 <- performance::r2(lmer_oracle)
    cat(sprintf("Oracle marginal  R² (lmer, fixed only):    %.3f\n", r2$R2_marginal))
    cat(sprintf("Oracle conditional R² (lmer, fixed + RE): %.3f\n", r2$R2_conditional))
  }
  cat("Random effect SDs from oracle lmer:\n")
  print(VarCorr(lmer_oracle))
} else {
  cat("lme4 not available — skipping lmer oracle check.\n")
}

# 3. Confirm true predictors stand out vs noise in marginal correlations
cors <- cor(Y, Z)
top5_idx <- order(abs(cors), decreasing = TRUE)[1:5]
cat(sprintf("\nTop-5 predictors by |cor(Y, Z)| : %s\n",
            paste(colnames(Z)[top5_idx], collapse = ", ")))
cat(sprintf("Correlations of true predictors V1–V5: %s\n",
            paste(round(cors[1:K_true], 3), collapse = ", ")))

cat("\nDataset generation complete.\n")
