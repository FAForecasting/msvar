% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.R
\name{predict.MSVAR}
\alias{predict.MSVAR}
\title{Create a forecast based on a msvar model}
\usage{
\method{predict}{MSVAR}(x, h, samples, Z_init = NULL)
}
\arguments{
\item{x}{The msvar model object}

\item{h}{The forecast horizon}

\item{samples}{The number of forecast samples to produce}

\item{Z_init}{The Z vector to use for prediction, defaults to the last part of Y if left as null}
}
\value{
An array of forecast samples
}
\description{
Create a forecast based on a msvar model
}
