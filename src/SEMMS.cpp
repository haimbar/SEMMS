#include <RcppArmadillo.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>

//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

//' Get an initial set of putative variables for the GAM algorithm
//'
//' @param Z matrix of putative variables
//' @param Yr the (possibly transformed) response
//' @param mincor a threshold of (absolute) correlation above which a pair is considered highly correlated
//' @return a list containing  variables to ignore because they are highly correlated with other, and SLR coefficients
// [[Rcpp::export]]
List initVals(NumericMatrix Z, arma::colvec Yr, double mincor=0.7) {
  int K = Z.ncol(), n = Z.nrow();
  NumericVector betas(K);
  IntegerVector discard(K);
  for (int i=0; i < K; i++) {
    NumericVector Z0 = Z(_, i);
    betas[i] = std::inner_product(Z0.begin(), Z0.end(), Yr.begin(), 0.0)/n;
    discard[i] = -1;
  }
  for (int i=0; i < K-1; i++) {
    NumericVector Z0 = Z(_, i);
    for (int j=i+1; j < K; j++) {
      if (discard[j] != -1) {
        continue;
      }
      NumericVector Z1 = Z(_, j);
      if (std::abs(std::inner_product(Z0.begin(), Z0.end(), Z1.begin(), 0.0))/n > mincor ) {
        if (std::abs(betas[i]) > std::abs(betas[j])) {
          discard[j] = i;
        } else {
          discard[i] = j;
        }
      }
    }
  }
  return List::create(Named("discard") = discard, Named("beta") = betas);
}


/* subset0 compares each element of a vector to 0 and returns
 * the index of the elements that are either greater than, less than,
 * not equal to, or equal to 0, depending on the value of op (>,<,!,=)
 */
IntegerVector subset0(IntegerVector x, char op) {
  int  n = x.size();
  std::vector<int> out(n, 0);
  int j = 0;
  for (int i = 0; i < n; i++) {
    if (((op == '>') && (x[i] > 0)) ||
        ((op == '<') && (x[i] < 0)) ||
        ((op == '!') && (x[i] != 0)) ||
        ((op == '=') && (x[i] == 0))) {
      out[j++] = i;
    }
  }
  out.resize(j);
  return wrap(out);
}

/* fetchZ extracts the columns in the vector nn from the matrix Z */
arma::mat fetchZ(IntegerVector nn,  NumericMatrix Z) {
  NumericVector Z0 = Z(_, nn[0]);
  arma::mat ZZ(Z0.begin(), Z0.size(), 1, false);
  for (int i = 1; i < nn.size(); i++){
    NumericVector Zi= Z(_, nn[i]);
    arma::mat ZZZ(Zi.begin(), Zi.size(), 1, false);
    ZZ = join_rows(ZZ, ZZZ );
  }
  return ZZ;
}

/* ZMatrix returns the matrix Z*Gamma */
arma::mat ZMatrix(IntegerVector nn, IntegerVector gamma, NumericMatrix Z){
  switch(nn.size()) {
  case 0: // no significant predictors — return N×0 empty matrix
    return arma::mat(Z.nrow(), 0);
  case 1: // just one significant predictor
    return fetchZ(nn, Z) * as<arma::colvec>(gamma[nn]);
  default:
    return fetchZ(nn, Z) * diagmat(as<arma::vec>(gamma[nn]));
  }
}


/* precisionMatrix: kept for reference; no longer called internally.
 * Internal code now uses Woodbury factors directly to avoid N×N ops.
 */
arma::mat precisionMatrix(double s2e, double s2r, arma::mat ZG, arma::mat W, int LT) {
  if (LT == 0)
    return(W/s2e);
  arma::mat ZGtW = ZG.t()*W;
  return(W/s2e - (s2r/pow(s2e,2)) *W * ZG *
         inv(arma::eye<arma::mat>(LT, LT) + (s2r/s2e)*ZGtW*ZG) * ZGtW);
}


/* loglik: compute log-likelihood via Woodbury identity — no N×N matrix.
 *
 * Σ = s2e·I_N + s2r·ZG·ZG'
 * Σ⁻¹ = (1/s2e)·I − (s2r/s2e²)·ZG·B·ZG'   where B = inv(I_LT + (s2r/s2e)·ZG'ZG)
 *
 * log|Σ⁻¹| = −N·log(s2e) − log|I_LT + (s2r/s2e)·ZG'ZG|   [det lemma]
 * yt'Σ⁻¹yt = ‖yt‖²/s2e − (s2r/s2e²)·w'·B·w   where w = ZG'·yt
 */
double loglik(arma::colvec Y, arma::mat X, IntegerVector L, double mu,
              arma::colvec beta, double s2r, double s2e, arma::mat ZGamma) {
  int LT = L[0] + L[2];
  int N  = (int)Y.n_rows;
  double ll = 0.0;
  for (int mm = 0; mm < 3; mm++) {
    if (L[mm] > 0)
      ll += (double)L[mm] * std::log((double)L[mm] /
            (double)(L[0]+L[1]+L[2]));
  }
  arma::colvec yt = Y - X * beta;
  if (LT > 0)
    yt -= ZGamma * (arma::ones<arma::vec>(LT) * mu);

  if (LT == 0) {
    double logdet = -(double)N * std::log(s2e);
    double quad   = arma::dot(yt, yt) / s2e;
    return ll - 0.5*quad + 0.5*logdet;
  }

  // LT×LT Woodbury factors
  arma::mat ZtZ = ZGamma.t() * ZGamma;
  arma::mat C   = arma::eye<arma::mat>(LT,LT) + (s2r/s2e) * ZtZ;
  arma::mat B   = arma::inv_sympd(C);

  // log|Σ⁻¹| = −N·log(s2e) − log|C|
  double logdetC, sign;
  arma::log_det(logdetC, sign, C);
  double logdet = -(double)N * std::log(s2e) - logdetC;

  // yt'Σ⁻¹yt via Woodbury: w = ZG'·yt (LT×1)
  arma::colvec w = ZGamma.t() * yt;
  double quad = arma::dot(yt, yt) / s2e
              - (s2r/(s2e*s2e)) * arma::as_scalar(w.t() * B * w);

  return ll - 0.5*quad + 0.5*logdet;
}


/* fit_gam: EM algorithm using Woodbury identity — no N×N matrices formed.
 *
 * Key pre-computations (fixed for a given ZGamma):
 *   ZtZ  = ZG'·ZG           (LT×LT)
 *   H    = [X | ZG]          (N×q,  q = xk+LT)
 *   HtH  = H'·H              (q×q)
 *   HtY  = H'·Yw             (q×1)
 *   HtZG = H'·ZG             (q×LT)
 *
 * Per EM step — only LT×LT inverse B needs rebuilding when s2e/s2r change:
 *   B  = inv(I_LT + (s2r/s2e)·ZtZ)
 *
 * s2e update uses: tr(s2e·I − s2e²·Σ⁻¹) = s2r·tr(ZtZ·B)  [simplifies exactly]
 */
List fit_gam(arma::colvec Yw, arma::mat X, NumericMatrix Z,
             double mu, arma::colvec beta,
             double s2r, double s2e, IntegerVector gam,
             double tol=1e-6, int maxsteps=20) {

  int xk = X.n_cols, n = (int)X.n_rows;
  double mu_old = mu, s2r_old = s2r, s2e_old = s2e, fc = 0.0;
  arma::colvec beta_old = beta;

  IntegerVector NullGrp = subset0(gam, '=');
  IntegerVector PosEff  = subset0(gam, '>');
  IntegerVector NegEff  = subset0(gam, '<');
  IntegerVector nn      = union_(PosEff, NegEff);
  std::sort(nn.begin(), nn.end());
  IntegerVector L(3);
  L[0] = NegEff.size();
  L[1] = NullGrp.size();
  L[2] = PosEff.size();
  int LT = L[0] + L[2];

  // LT == 0: trivial case, no N×N issues
  if (LT == 0) {
    arma::colvec yt = Yw - X * beta;
    double logdet = -(double)n * std::log(s2e);
    double quad   = arma::dot(yt, yt) / s2e;
    return List::create(Named("mu")=0.0,
                        Named("beta")=arma::solve(X.t()*X, X.t()*Yw),
                        Named("s2r")=1e-5,
                        Named("s2e")=arma::var(Yw),
                        Named("ll")=(-0.5*quad + 0.5*logdet));
  }

  arma::mat ZGamma = ZMatrix(nn, gam, Z);   // N×LT

  // ── Pre-compute quantities fixed for this call to fit_gam ────────────────
  int q = xk + LT;
  arma::mat H     = arma::join_rows(X, ZGamma);   // N×q
  arma::mat ZtZ   = ZGamma.t() * ZGamma;          // LT×LT
  arma::mat HtH   = H.t() * H;                    // q×q
  arma::colvec HtY = H.t() * Yw;                  // q×1
  arma::mat HtZG  = H.t() * ZGamma;               // q×LT  (= [X'ZG ; ZtZ])
  // ZtY = last LT rows of HtY, but extract explicitly for clarity
  arma::colvec ZtY = ZGamma.t() * Yw;             // LT×1
  // ─────────────────────────────────────────────────────────────────────────

  double totalerr = 1.0;
  int count = 0;
  while (totalerr > tol) {
    if (++count >= maxsteps) break;

    // ── Build Woodbury factors for current (s2e, s2r) ─────────────────────
    double coef1 = s2r / (s2e * s2e);
    arma::mat C1  = arma::eye<arma::mat>(LT,LT) + (s2r/s2e) * ZtZ;
    arma::mat B1  = arma::inv_sympd(C1);                // LT×LT

    // H'Σ⁻¹H = H'H/s2e − coef1·HtZG·B1·HtZG'
    arma::mat HtSiH = HtH/s2e - coef1 * HtZG * (B1 * HtZG.t());   // q×q

    // H'Σ⁻¹Y = H'Y/s2e − coef1·HtZG·(B1·ZtY)
    arma::colvec HtSiY = HtY/s2e - coef1 * HtZG * (B1 * ZtY);     // q×1

    // Solve for theta (more stable than explicit inverse)
    arma::colvec thetav = arma::solve(HtSiH, HtSiY);               // q×1

    mu   = arma::median(arma::abs(thetav.subvec(xk, q-1)));
    for (int ii = xk; ii < q; ii++) thetav[ii] = mu;
    beta = thetav.subvec(0, xk-1);

    // ── t_e update (s2e) ──────────────────────────────────────────────────
    arma::colvec Y0  = Yw - H * thetav;          // N×1
    arma::colvec w0  = ZGamma.t() * Y0;          // LT×1
    arma::colvec Bw0 = B1 * w0;                  // LT×1

    // Σ⁻¹·Y0 = Y0/s2e − coef1·ZGamma·Bw0      (N×1, formed explicitly)
    arma::colvec SiY0 = Y0/s2e - coef1 * ZGamma * Bw0;

    // tr(s2e·I − s2e²·Σ⁻¹) = s2r·tr(ZtZ·B1)   [exact simplification]
    double t_e = s2r * arma::trace(ZtZ * B1)
               + s2e*s2e * arma::dot(SiY0, SiY0);
    s2e = t_e / (double)n;
    if (s2e <= 0.0) s2e = 1e-10;

    // ── Rebuild B for updated s2e ─────────────────────────────────────────
    double coef2 = s2r / (s2e * s2e);
    arma::mat C2  = arma::eye<arma::mat>(LT,LT) + (s2r/s2e) * ZtZ;
    arma::mat B2  = arma::inv_sympd(C2);                // LT×LT

    // ── t_r update (s2r) ──────────────────────────────────────────────────
    // ZG'Σ⁻¹ZG = ZtZ/s2e − coef2·ZtZ·B2·ZtZ   (LT×LT)
    arma::mat ZtSiZ = ZtZ/s2e - coef2 * ZtZ * B2 * ZtZ;

    // tr(s2r_old·I_LT − s2r_old²·ZG'Σ⁻¹ZG)
    double trace_r = s2r_old * (double)LT
                   - s2r_old*s2r_old * arma::trace(ZtSiZ);

    // ZG'·Σ⁻¹·Y0 = w0/s2e − coef2·ZtZ·(B2·w0)   (LT×1)
    arma::colvec Bw0_2  = B2 * w0;
    arma::colvec ZtSiY0 = w0/s2e - coef2 * ZtZ * Bw0_2;

    double t_r = trace_r + s2r_old*s2r_old * arma::dot(ZtSiY0, ZtSiY0);
    s2r = t_r / (double)LT;
    if (s2r <= 0.0) s2r = 1e-10;

    if (mu > 2.0*std::sqrt(s2e))
      s2r = std::max(std::pow((mu - 2.0*std::sqrt(s2e))/3.0, 2.0), s2r);

    // ── Convergence ───────────────────────────────────────────────────────
    fc = loglik(Yw, X, L, mu, beta, s2r, s2e, ZGamma);

    totalerr = std::pow(mu  - mu_old,  2)
             + std::pow(s2e - s2e_old, 2)
             + std::pow(s2r - s2r_old, 2)
             + arma::as_scalar(arma::sum(arma::pow(beta - beta_old, 2)));

    mu_old   = mu;
    beta_old = beta;
    s2r_old  = s2r;
    s2e_old  = s2e;
  }

  return List::create(Named("mu")=mu, Named("beta")=beta,
                      Named("s2r")=s2r, Named("s2e")=s2e, Named("ll")=fc);
}


//' Run the GAM algorithm to select non-null variables
//'
//' @param initidx an initial set of variables to use in the fitting algorithm
//' @param initval the values (-1 or 1) of the initial set of predictors
//' @param Yr the response vector
//' @param Xr the fixed-effect design matrix
//' @param Z the matrix of all putative variables
//' @param distr the GLM distribution to fit (N=Normal, B=binary, P=Poisson)
//' @param randomize Boolean - whether to run the greedy or randomized version
//' @param mincor a threshold of (absolute) correlation above which a pair is considered highly correlated
//' @param maxsteps maximum number of GAM iterations
//' @param minchange the minimum difference in log-likelihood between consecutive iterations below which we assume that the algorithm has converged
//' @param ptf Boolean - whether to print debug messages to SEMMS.log
//' @return a list containing the index of non-null variables (columns in Z), the mixture model parameters, the sign of the selected coefficients, a matrix with posterior probabilities, and an indicator array for locked out variables
// [[Rcpp::export]]
List GAMupdate(IntegerVector initidx, IntegerVector initval, arma::colvec Yr, NumericMatrix Xr,
               NumericMatrix Z, char distr = 'N', bool randomize=true, double mincor=0.7,
               int maxsteps=20, double minchange= 1, bool ptf = true ) {
  FILE *ofp;
  if (ptf) {
    char outputFilename[] = "SEMMS.log";
    ofp = fopen(outputFilename, "a");
    if (ofp == NULL) {
      Rcpp::stop("Can't open output file SEMMS.log!");
    }
    fprintf(ofp,"==========\n");
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    fprintf(ofp, "%d-%d-%d %d:%d:%d\n", tm.tm_year + 1900, tm.tm_mon + 1,
            tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  }

  int n = Xr.nrow(), xk = Xr.ncol(), K = Z.ncol();
  arma::mat X(Xr.begin(), n, xk, false);
  arma::colvec Y  = Yr;
  arma::colvec Yw = Yr;           // working response

  // No N×N identity matrix needed — removed
  NumericMatrix pp(K, 3);
  IntegerVector gam(K);
  gam[initidx] = initval;

  double mu = 0.0, s2r = var(Yr)/10, s2e = var(Yr)/2, ll = 0.0,
         p0 = 1.0, p1 = 1.0, p2 = 1.0, psum = 3.0, CK = 0.0;
  if (distr == 'P') {
    Yw  = log(Y+0.1);
    s2e = var(Yw)/2;
    s2r = s2e/10;
    mu  = max(abs(Yw-mean(Yw)))*0.9;
  }
  if (distr == 'B') {
    Yw  = log((Y+0.1)/(1-Y+0.1));
    s2e = var(Yw)/2;
    s2r = s2e/10;
    mu  = max(abs(Yw-mean(Yw)))*0.9;
  }

  arma::colvec beta(xk);
  NumericVector f0pn(3);
  IntegerVector gam_tmp(gam.size()), nnt(1), Component(3);
  IntegerVector idx(1);
  arma::mat ZGammat, zzz(1,1), cr(1,1);   // ZGammat declared without N×K pre-allocation
  IntegerVector lockOut(K);
  bool kNotIncluded = true;
  if (randomize) srand((unsigned)time(NULL));

  for (int j = 0; j < maxsteps; j++) {
    if (ptf) fprintf(ofp, "Iteration %d\n", j);

    List fitted = fit_gam(Yw, X, Z, mu, beta, s2r, s2e, gam);
    mu   = fitted["mu"];
    beta = as<arma::colvec>(fitted["beta"]);
    s2r  = fitted["s2r"];
    s2e  = fitted["s2e"];
    ll   = fitted["ll"];


    NumericVector delta_ll(K, 0.0);
    IntegerVector newgamk_array(K);
    idx = subset0(gam, '!');

    for (int k = 0; k < K; k++) {
      CK = 0.0;
      kNotIncluded = true;
      IntegerVector nnn(idx.size()+1);
      for (int i = 0; i < idx.size(); i++) {
        nnn[i] = idx[i];
        if (idx[i] == k) {
          kNotIncluded = false;
          break;
        }
      }
      gam_tmp = clone(gam);
      for (int mm = 0; mm < 3; mm++) {
        gam_tmp[k] = mm-1;
        if (gam_tmp[k] == gam[k]) {
          f0pn[mm] = ll;
        } else {
          nnt = subset0(gam_tmp, '!');
          if (nnt.length() == 0) {
            // No selected variables: Σ = s2e·I, log|Σ⁻¹| = -N·log(s2e)
            arma::colvec yt = Yw - X * beta;
            double logdet = -(double)n * std::log(s2e);
            double quad   = arma::dot(yt, yt) / s2e;
            f0pn[mm] = -0.5*quad + 0.5*logdet;
          } else {
            ZGammat    = ZMatrix(nnt, gam_tmp, Z);
            Component[0] = subset0(gam_tmp, '<').size();
            Component[1] = subset0(gam_tmp, '=').size();
            Component[2] = subset0(gam_tmp, '>').size();
            f0pn[mm] = loglik(Yw, X, Component, mu, beta, s2r, s2e, ZGammat);
          }
        }
      }

      p1   = subset0(gam, '<').size() * exp(f0pn[0]);
      p0   = subset0(gam, '=').size() * exp(f0pn[1]);
      p2   = subset0(gam, '>').size() * exp(f0pn[2]);
      psum = p0 + p1 + p2;
      if (psum > 0.0) {
        pp(k,0) = p0/psum;
        pp(k,1) = p1/psum;
        pp(k,2) = p2/psum;
      } else {
        pp(k,0) = pp(k,1) = pp(k,2) = 1.0/3.0;
      }

      // Lock out variables highly correlated with currently-selected ones
      if (kNotIncluded && (idx.size() > 0)) {
        nnn[idx.size()] = k;
        zzz = fetchZ(nnn, Z);
        cr  = abs(cor(zzz));
        CK  = cr.submat(idx.size(), 0, idx.size(), idx.size()-1).max();
        if (CK > mincor) {
          lockOut[k] = 1;
          continue;
        } else {
          lockOut[k] = 0;
        }
      }

      if (max(f0pn) - ll > minchange) {
        if (ptf) fprintf(ofp, "%d %g, %g, %g\n", k, f0pn[0], f0pn[1], f0pn[2]);
        delta_ll[k]       = max(f0pn) - ll;
        newgamk_array[k]  = which_max(f0pn) - 1;
      }
    }

    if (max(delta_ll) > 0) {
      int chosen_k = 0;
      if (randomize) {
        NumericVector relativeweights = 100 * delta_ll / sum(delta_ll);
        int thr = rand() % 100;
        double csum = 0;
        for (int ii = 0; ii < delta_ll.length(); ii++) {
          csum += relativeweights[ii];
          if (csum > thr) { chosen_k = ii; break; }
        }
      } else {
        chosen_k = which_max(delta_ll);
      }
      int gg = gam[chosen_k];
      gam[chosen_k] = newgamk_array[chosen_k];
      ll = delta_ll[chosen_k] + ll;
      if (ptf)
        fprintf(ofp, "Changing  %d from %d to %d \n",
                chosen_k, gg, newgamk_array[chosen_k]);
    } else {
      break;
    }

    // Update working response for GLM families
    IntegerVector nn2 = union_(subset0(gam, '>'), subset0(gam, '<'));
    arma::colvec eta  = X * beta;
    if (nn2.length() != 0) {
      std::sort(nn2.begin(), nn2.end());
      arma::mat ZGamma2 = ZMatrix(nn2, gam, Z);
      arma::colvec muvec = arma::ones<arma::vec>(nn2.length()) * mu;
      eta = eta + ZGamma2 * muvec;
    }
    if (distr == 'P') Yw = eta - 1 + Yr % (exp(-eta));
    if (distr == 'B') Yw = eta - exp(eta) - 1 + Yr % (1/exp(eta) + 2 + exp(eta));
  }

  IntegerVector nn = subset0(gam, '!');
  if (ptf) {
    fprintf(ofp, "Selected: ");
    for (int i = 0; i < nn.length(); i++) fprintf(ofp, "%d ", nn[i]+1);
    fprintf(ofp, "\n\n");
    fclose(ofp);
  }
  return List::create(Named("nn")      = nn+1,
                      Named("mu")      = mu,
                      Named("beta")    = beta,
                      Named("s2r")     = s2r,
                      Named("s2e")     = s2e,
                      Named("gam_nn")  = gam[nn],
                      Named("pp")      = pp,
                      Named("lockedOut") = lockOut);
}
