#include <RcppArmadillo.h>
#include <time.h>
#include <stdlib.h>

//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List initVals(NumericMatrix Z, arma::colvec Yr, float mincor=0.7) {
  int K = Z.ncol(), n = Z.nrow();
  NumericVector betas(K);
  IntegerVector discard(K);
  arma::mat zzz(n,2);
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
  arma::mat ZG(Z.ncol(),1);
  switch(nn.size()) {
  case 0: // no significant predictors
    return ZG;
  case 1: // just one significant predictor
    return fetchZ(nn, Z) * as<arma::colvec>(gamma[nn]);
  default:
    return fetchZ(nn, Z) * diagmat(as<arma::vec>(gamma[nn]));
  }
}


/* Calculate the precision matrix for the multivariate normal
 * distribution.
 */
arma::mat precisionMatrix(double s2e, double s2r, arma::mat ZG, arma::mat W, int LT) {
  if (LT == 0)    // no significant predictors
    return(W/s2e);
  arma::mat ZGtW = ZG.t()*W;
  return(W/s2e - (s2r/pow(s2e,2)) *W * ZG *
         inv(arma::eye<arma::mat>(LT, LT) + (s2r/s2e)*ZGtW*ZG) * ZGtW);
}


/* Calculate the log-likelihood using only the selected variables */
double loglik(arma::colvec Y, arma::mat X, IntegerVector L, double mu, arma::colvec beta,
              double s2r, double s2e, arma::mat ZGamma, arma::mat W) {
  int LT = L[0] + L[2];
  double ll = 0; // -0.5*(double)N*log(2*s2e*datum::pi);
  for (int mm = 0; mm < 3; mm++) {
    if (L[mm] > 0) {
      ll += (double)L[mm]*log((double)L[mm] / (double)(L[0]+L[1]+L[2]));
    }
  }
  arma::mat SigmaInv = precisionMatrix(s2e, s2r, ZGamma, W, LT);
  double val, sign;
  log_det(val, sign, SigmaInv);
  arma::colvec yt(Y);
  yt = Y - X * beta;
  if (LT > 0) {
    arma::colvec muvec = arma::ones<arma::vec>(LT)*mu;
    yt = yt - ZGamma * muvec;
  }
  return(as_scalar(ll - 0.5* yt.t()*SigmaInv*yt + 0.5*val));
}

/* fit_gam returns the fitted parameters (using the EM algorithm)
 * and the updated loglikelihood */
List fit_gam(arma::colvec Y, arma::mat X, NumericMatrix Z, double mu, arma::colvec beta,
            double s2r, double s2e, IntegerVector gam, arma::mat W,
            double tol=1e-6, int maxsteps=20) {
  int xk = X.n_cols, n = X.n_rows;
  double mu_old = mu,  s2r_old =  s2r, s2e_old = s2e,
    fc_old = -1000.0, fc = 0.0;
  arma::colvec beta_old = beta,  t_e(1), t_r(1), theta(1);
  arma::mat I_N = arma::eye<arma::mat>(n,n), H(n,xk+1), SigmaInv(1,1);
  IntegerVector NullGrp = subset0(gam, '=');
  IntegerVector PosEff =  subset0(gam, '>');
  IntegerVector NegEff =  subset0(gam, '<');
  IntegerVector nn = union_(PosEff, NegEff);
  std::sort(nn.begin(), nn.end());
  IntegerVector L(3);
  L[0] = NegEff.size();
  L[1] = NullGrp.size();
  L[2] = PosEff.size();
  int LT  = L[0] + L[2];
  if (LT == 0) {
    double val, sign;
    log_det(val, sign, W/s2e);
    arma::colvec yt(Y);
    yt = Y - X * beta;
    return List::create(Named("mu")=0.0, Named("beta")=inv(X.t() * X) * X.t()*Y,
                        Named("s2r")=1e-5, Named("s2e")=var(Y),
                        Named("ll")=as_scalar(-0.5* yt.t()*(W/s2e)*yt + 0.5*val));
  }
  arma::mat ZGamma = ZMatrix(nn, gam, Z);
  double totalerr  = 1;
  int count = 0;
  while (totalerr > tol){
    count++;
    if (count >= maxsteps) {
      break;
    }

    H = join_rows(X,ZGamma);
    SigmaInv = precisionMatrix(s2e, s2r, ZGamma, W, LT);
    arma::mat HtSi = H.t() * SigmaInv;
    theta = inv(HtSi * H) * HtSi * Y;
    mu = median(abs(theta.subvec(xk,theta.n_rows-1)));
    for (int ii=xk; ii<theta.n_rows-1; ii++)
      theta[ii] =  mu;
    beta =  theta.subvec(0, xk-1);
    arma::mat Y0tSi = (Y-H*theta).t()*SigmaInv;
    t_e = sum(arma::mat (s2e_old*I_N - pow(s2e_old, 2) * SigmaInv).diag()) +
      pow(s2e_old,2)*Y0tSi*Y0tSi.t();
    s2e = t_e[0]/n;
    SigmaInv = precisionMatrix(s2e, s2r, ZGamma, W, LT);
    arma::mat Y0tSiZG = (Y-H*theta).t()*SigmaInv*ZGamma;
    t_r =  sum(arma::mat (s2r_old*arma::eye<arma::mat>(LT,LT) -
      pow(s2r_old,2) * ZGamma.t() * SigmaInv * ZGamma).diag()) +
      pow(s2r_old,2) * Y0tSiZG * Y0tSiZG.t();
    s2r = t_r[0]/LT;
    if (mu > 2*sqrt(s2e)) {
      s2r = std::max(pow((mu-2*sqrt(s2e))/3.0,2),s2r);
    }

    fc = loglik(Y, X, L, mu, beta, s2r, s2e, ZGamma, I_N);
    totalerr = pow(mu - mu_old, 2) + pow(s2e - s2e_old, 2) +
      pow(s2r - s2r_old, 2) + sum(pow(beta - beta_old,2));
    mu_old = mu;
    beta_old = beta;
    s2r_old = s2r;
    s2e_old = s2e;
    fc_old  = fc;
  }
  return List::create(Named("mu")=mu, Named("beta")=beta,
                      Named("s2r")=s2r, Named("s2e")=s2e,
                      Named("ll")=fc);
}

// [[Rcpp::export]]
List GAMupdate(IntegerVector initidx, IntegerVector initval, arma::colvec Yr, NumericMatrix Xr,
              NumericMatrix Z, char distr = 'N', bool randomize=true, double mincor=0.7,
              int maxsteps=20, double minchange= 1, bool ptf = true ) {
  FILE *ofp;
  if (ptf) {
    char outputFilename[] = "SEMMS.log";
    ofp = fopen(outputFilename, "a");
    if (ofp == NULL) {
      fprintf(stderr, "Can't open output file %s!\n",
              outputFilename);
      exit(1);
    }
    fprintf(ofp,"==========\n");
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    fprintf(ofp, "%d-%d-%d %d:%d:%d\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  }
  int n = Xr.nrow(), xk=Xr.ncol(), K = Z.ncol();
  arma::mat X(Xr.begin(), n, xk, false);
  arma::colvec Y = Yr;
  arma::colvec Yw = Yr;           // working response
  arma::mat I_N = arma::eye<arma::mat>(n,n);  // weight matrix
  NumericMatrix pp(K, 3);
  IntegerVector gam(K);
  gam[initidx] = initval;
  double mu= 0.0, s2r = var(Yr)/10, s2e = var(Yr)/2, ll = 0.0,
    p0 = 1.0, p1 = 1.0, p2 = 1.0, psum = 3.0, CK = 0.0, maxc = 0.0;
  if (distr == 'P') {
    Yw = log(Y+0.1);
    s2e = var(Yw)/2;
    s2r = s2e/10;
    mu  = max(abs(Yw-mean(Yw)))*0.9;
  }
  if (distr == 'B') {
    Yw = log((Y+0.1)/(1-Y+0.1));
    s2e = var(Yw)/2;
    s2r = s2e/10;
    mu  = max(abs(Yw-mean(Yw)))*0.9;
  }
  arma::colvec beta(xk);
  NumericVector f0pn(3), p(3), kk(1);
  IntegerVector gam_tmp(gam.size()), nnt(1), Component(3);
  IntegerVector idx(1);
  arma::mat ZGammat(n,K),zzz(1,1),cr(1,1), H(n,xk+1);
  IntegerVector lockOut(K);
  bool kNotIncluded = true;
  for (int j = 0; j < maxsteps; j++) {
    if (ptf) {
      fprintf(ofp, "Iteration %d\n",j);
    }
    List fitted = fit_gam(Yw,X,Z, mu, beta, s2r,s2e, gam, I_N);
    mu = fitted["mu"];
    beta = as<arma::colvec>(fitted["beta"]);
    s2r = fitted["s2r"];
    s2e = fitted["s2e"];
    ll = fitted["ll"];
    maxc = minchange;
    NumericVector delta_ll(K,0.0);
    IntegerVector int_delta_ll(K,0);
    IntegerVector newgamk_array(K);
    idx = subset0(gam, '!');
    for (int k = 0; k < K; k++) {
      CK = 0.0;
      kNotIncluded = true;
      IntegerVector nnn(idx.size()+1);
      for(int i = 0; i < idx.size(); i++){
        nnn[i] = idx[i];
        if (idx[i] == k) {
          kNotIncluded = false;
          break;
        }
      }
      gam_tmp = clone(gam);
      for (int mm = 0; mm < 3; mm++){
        gam_tmp[k] = mm-1;
        if (gam_tmp[k] == gam[k]) {
          f0pn[mm] = ll;
        } else {
          nnt = subset0(gam_tmp, '!');
          if (nnt.length() == 0) {
            double val, sign;
            log_det(val, sign, I_N/s2e);
            arma::colvec yt(Yw);
            yt = Yw - X * beta;
            f0pn[mm] = as_scalar(-0.5* yt.t()*(I_N/s2e)*yt + 0.5*val);
          } else {
            ZGammat = ZMatrix(nnt, gam_tmp, Z);
            Component[0] = subset0(gam_tmp, '<').size();
            Component[1] = subset0(gam_tmp, '=').size();
            Component[2] = subset0(gam_tmp, '>').size();
            f0pn[mm] = loglik(Yw, X, Component, mu, beta, s2r, s2e, ZGammat, I_N);
          }
        }
      }
      p1 = subset0(gam, '<').size()*exp(f0pn[0]);
      p0 = subset0(gam, '=').size()*exp(f0pn[1]);
      p2 = subset0(gam, '>').size()*exp(f0pn[2]);
      psum = p0 + p1 + p2;
      pp(k,0) = p0/psum;
      pp(k,1) = p1/psum;
      pp(k,2) = p2/psum;
      // if k-th variable is not already in the model, check if it's
      // highly correlated with a variable in the model:
      if (kNotIncluded && (idx.size() > 0)) {
        nnn[idx.size()] = k;
        zzz = fetchZ(nnn,Z);
        cr = abs(cor(zzz));
        CK = cr.submat( idx.size() , 0, idx.size(), idx.size()-1 ).max();
        if (CK > mincor) {
          lockOut[k] = 1;
          continue;
        }
        else {
          lockOut[k] = 0;
        }
      }
      if (max(f0pn)-ll > minchange) {
        if (ptf) {
          fprintf(ofp, "%d %g, %g, %g\n",k, f0pn[0],f0pn[1],f0pn[2]);
        }
        delta_ll[k] = max(f0pn) - ll;
        int_delta_ll[k] = 1;
        newgamk_array[k] = which_max(f0pn)-1;
      }
    }

    if (max(delta_ll) > 0) {
      int chosen_k = 0;
      if (randomize) {
        NumericVector relativeweights = 100*delta_ll/sum(delta_ll);
        srand(time(NULL));
        int thr = rand() % 100;
        double csum = 0;
        for (int ii = 0; ii < delta_ll.length(); ii++) {
          csum += relativeweights[ii];
          if (csum > thr) {
            chosen_k = ii;
            break;
          }
        }
      } else {
        chosen_k = which_max(delta_ll);
      }
      int gg = gam[chosen_k];
      gam[chosen_k] = newgamk_array[chosen_k];
      ll = delta_ll[chosen_k] + ll;
      if (ptf) {
        fprintf(ofp,"Changing  %d from %d to %d \n",chosen_k, gg, newgamk_array[chosen_k]);
      }
    } else {
      break;
    }
    /*
     * with the updated vector of nonnull indicator variables,
     * obtain the updated weight matrix, and the working response
     */
    IntegerVector nn = union_(subset0(gam, '>'), subset0(gam, '<'));
    arma::colvec eta = X*beta;
    if (nn.length()!=0){
      std::sort(nn.begin(), nn.end());
      arma::mat ZGamma = ZMatrix(nn, gam, Z);
      arma::colvec muvec = arma::ones<arma::vec>(nn.length())*mu;
      eta = eta + ZGamma*muvec;
    }
    if (distr == 'P') {
      Yw = eta - 1 + Yr%(exp(-eta));
    }
    if (distr == 'B') {
      Yw = eta-exp(eta) - 1 + Yr%(1/exp(eta)+2+exp(eta));
    }
  }
  IntegerVector nn = subset0(gam, '!');
  if (ptf) {
    fprintf(ofp,"Selected: ");
    for (int i=0; i < nn.length(); i++) {
      fprintf(ofp,"%d ",nn[i]+1);
    }
    fprintf(ofp,"\n\n");
    fclose(ofp);
  }
  return List::create(Named("nn") = nn+1, Named("mu") = mu,
                      Named("beta") = beta, Named("s2r") = s2r,
                      Named("s2e") = s2e, Named("gam_nn") = gam[nn],
                                                               Named("pp") = pp, Named("lockedOut") = lockOut);
}

