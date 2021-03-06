% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{GAMupdate}
\alias{GAMupdate}
\title{Get an initial set of putative variables for the GAM algorithm}
\usage{
GAMupdate(
  initidx,
  initval,
  Yr,
  Xr,
  Z,
  distr = "N",
  randomize = TRUE,
  mincor = 0.7,
  maxsteps = 20L,
  minchange = 1,
  ptf = TRUE
)
}
\arguments{
\item{initidx}{an initial set of variables to use in the fitting algorithm}

\item{initval}{the values (-1 or 1) of the initial set of preditors}

\item{Yr}{the response vector}

\item{Xr}{the fixed-effect design matrix}

\item{Z}{the matrix of all putative variables}

\item{distr}{the GLM distribution to fit (N=Normal, B=binary, P=Poisson)}

\item{randomize}{Boolean - whether to run the greedy or randomized version}

\item{mincor}{a threshold of (absolute) correlation above which a pair is considered highly correlated}

\item{maxsteps}{maximum number of GAM iterations}

\item{minchange}{the minimum difference in log-likelihood between consecutive iterations below which we assume that the algorithm has  converged}

\item{ptf}{Boolean - whether to print debug messages to SEMMS.log}
}
\value{
a list containing  the index of non-null variables (columns in Z),  the mixture model parameters, the sign of the selected coefficients, a matrix with posterior probabilities, and an indicator array for locked out variables
}
\description{
Get an initial set of putative variables for the GAM algorithm
}
