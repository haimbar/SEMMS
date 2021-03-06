% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/semmsFuncs.R
\docType{data}
\name{AR1SIM}
\alias{AR1SIM}
\title{Simulated gene expression data}
\format{
A 1001 by 100 matrix.
}
\description{
AR1SIM is a simulated dataset with a hub structure, consisting of 1000 predictors arranged in 50 hubs. The response variable was constructed as the sum of the variables in the first hub, plus some Gaussian noise.
}
\examples{
\dontrun{
fn <- system.file("extdata", "AR1SIM.RData", package = "SEMMS", mustWork = TRUE)
dataYXZ <- readInputFile(fn, ycol=1, Zcols=2:100)
fittedSEMMS <- fitSEMMS(dataYXZ, mincor=0.8, nn=15, minchange= 1,
                        distribution="N",verbose=T,rnd=F)
fittedGLM <- runLinearModel(dataYXZ,fittedSEMMS$gam.out$nn, "N")
plotMDS(dataYXZ, fittedSEMMS, fittedGLM, ttl="AR1 simulation")}
}
\keyword{datasets}
