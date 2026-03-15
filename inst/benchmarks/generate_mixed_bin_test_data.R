# generate_mixed_bin_test_data.R
#
# Simulate a longitudinal dataset with a binomial response for testing
# fitSEMMSmixed() with distribution = "B".
#
# Design:
#   N_subj = 30, N_time = 4  =>  N = 120 observations
#   K = 100 candidate predictors (standardised at read time)
#   K_true = 5  (columns 1-5 are truly associated)
#   Random intercept (sigma_b0 = 1.0) + random slope over time (sigma_b1 = 0.5)
#
# Linear predictor (logit scale):
#   eta_ij = Z[,1:5] %*% beta_true + b0_i + b1_i * time_sc_ij
#   Y_ij   ~ Bernoulli( logistic(eta_ij) )
#
# Column layout in saved data frame:
#   col 1 : Y        (binary 0/1 response)
#   col 2 : subject  (factor, 1-30)
#   col 3 : time     (integer, 1-4)
#   col 4-103 : V1-V100 (standardised predictors)
#
# Reproducibility: set.seed(20260314)
# Output: inst/extdata/sim_mixed_bin_N120_P100_k5.RData

set.seed(20260314)

N_subj  <- 30L
N_time  <- 4L
N       <- N_subj * N_time     # 120
K       <- 100L
K_true  <- 5L

beta_true <- c(1.5, -1.3, 1.1, -1.0, 0.9)
sigma_b0  <- 1.0
sigma_b1  <- 0.5

subject <- rep(seq_len(N_subj), each  = N_time)
time    <- rep(seq_len(N_time), times = N_subj)
time_sc <- as.numeric(scale(time))

Z  <- matrix(rnorm(N * K), N, K)
colnames(Z) <- paste0("V", seq_len(K))

b0 <- rnorm(N_subj, 0, sigma_b0)
b1 <- rnorm(N_subj, 0, sigma_b1)

eta <- as.numeric(Z[, seq_len(K_true)] %*% beta_true) +
       b0[subject] + b1[subject] * time_sc

p   <- 1 / (1 + exp(-eta))
Y   <- rbinom(N, 1, p)

sim_df <- data.frame(
  Y       = Y,
  subject = factor(subject),
  time    = time,
  Z
)

cat(sprintf("N=%d (%d subj x %d time)  K=%d  K_true=%d\n",
            N, N_subj, N_time, K, K_true))
cat(sprintf("Mean(Y)=%.3f  sigma_b0=%.1f  sigma_b1=%.1f\n",
            mean(Y), sigma_b0, sigma_b1))

out_file <- file.path("..", "extdata", "sim_mixed_bin_N120_P100_k5.RData")
save(sim_df, file = out_file)
cat("Saved to", normalizePath(out_file), "\n")
