# SEMMS 0.4.1

## New features

* `fitSEMMSmixed()` now supports **Poisson** (`distribution = "P"`) and
  **binomial** (`distribution = "B"`) responses in addition to Gaussian.
  The fixed-effects step uses the proper Penalised Quasi-Likelihood (PQL)
  IRLS working response (Breslow & Clayton, 1993):

  - Poisson (log link):    `W* = η̂_fixed + (Y − μ̂) / μ̂`
  - Binomial (logit link): `W* = η̂_fixed + (Y − π̂) / (π̂(1 − π̂))`

  Because `μ̂` and `π̂` are continuous fitted values from `glmer`, the
  working response is continuous at every outer iteration for both families.

* Added `.semms_working_response()` (internal): computes the RE-adjusted
  IRLS working response from a fitted `lmer`/`glmer` object for any of the
  three supported families.

* `.semms_fit_mixed()` (internal): unified `lmer`/`glmer` fitter used
  throughout the alternating loop; dispatches to `lmer` for Gaussian and
  `glmer` (with the appropriate family object) for Poisson/binomial.

## Documentation

* Vignette updated: removed the "Gaussian only" caveat, added description
  of the IRLS working response, added non-Gaussian usage examples, and
  added the Breslow & Clayton (1993) reference.
* `DESCRIPTION` updated to reflect non-Gaussian support.

---

# SEMMS 0.3.0

## Bug fixes

* `fitSEMMS`: `scale()` now correctly returns a vector via `as.vector()`, preventing downstream dimension mismatches.
* `fitSEMMS`: Fixed a parenthesis bug in the `initWithEdgeFinder` branch where `length(which(...) > 0)` was always `TRUE`; corrected to `length(which(...)) > 0`.
* `mds2D`: `sqrt()` of potentially negative eigenvalues now uses `sqrt(abs(...))`, preventing `NaN` coordinates on near-singular distance matrices.
* `readInputFile`: Added an informative error for unrecognized file extensions instead of silently failing.
* `ZMatrix` (C++): The empty-selection case now returns an `N×0` matrix instead of an uninitialized `K×1` matrix.
* `fit_gam` (C++): Added positivity floors (`1e-10`) on `s2e` and `s2r` after EM updates to prevent degenerate variance estimates.
* `GAMupdate` (C++): Posterior probability computation is now guarded against division by zero (`psum == 0`).
* `GAMupdate` (C++): `srand()` is now called once before the main loop instead of being re-seeded on every iteration.
* `GAMupdate` (C++): `exit(1)` on log-file open failure replaced with `Rcpp::stop()` to avoid crashing the R session.

## Performance

* **~38× speedup** on large datasets (N=500, P=1,000): `loglik` and `fit_gam` no longer form N×N matrices.
  - Log-determinant: O(N³) → O(L³) via the matrix determinant lemma, where L is the number of currently-selected variables (L << N).
  - Quadratic forms and traces: O(N²) → O(N·L) via the Woodbury identity.
  - `fit_gam` pre-computes `H`, `ZtZ`, `HtH`, `HtY`, and `HtZG` once per call (fixed across EM iterations).
  - `arma::solve` replaces explicit `inv(A)*b` for improved numerical stability.

## Documentation

* Corrected `mincor` default in `fitSEMMS` documentation (was 0.75, is 0.7).
* Removed incorrect note that `mincor` is ignored when `initWithEdgeFinder = TRUE` — it is always used as the correlation lockout threshold in `GAMupdate`.
* Removed spurious `Z` element from `runLinearModel` return-value documentation.
* Corrected `AR1SIM` dataset `@format` (dimensions were transposed).
* Updated all AR1SIM-based examples to use the full predictor range (`Zcols=2:1001`).

## Code quality

* `initVals` (C++): `mincor` parameter type changed from `float` to `double` for consistency.
* `GAMupdate` (C++): Removed unused variables `p`, `kk` (C++), and `zzz` (in `initVals`).

---

# SEMMS 0.2.0

* Added `initWithEdgeFinder` option: uses the `edgefinder` package to identify highly correlated predictor clusters and select cluster centres as the initial candidate set.
* `plotMDS` now shows variables that are locked out due to high correlation with selected predictors.

---

# SEMMS 0.1.1

* Response and predictors are now scaled automatically.
* Added a warning when no predictors survive the Benjamini-Hochberg adjustment during initialization.
* Removed dependency on the `smacof` package; `mds2D` is now implemented internally.
