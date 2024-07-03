#' Create a forecast based on a msvar model
#' @param x The msvar model object
#' @param h The forecast horizon
#' @param samples The number of forecast samples to produce
#' @param Z_init The Z vector to use for prediction, defaults to the last part of Y if left as null
#' @return An array of forecast samples
#' @method predict MSVAR
#' @export
predict.MSVAR <- function (x, h, samples, Z_init = NULL) {
  state_vec <- seq_len(x$h)

  # Decompose sigma matrices for mvnorm draws
  sigmaU <- lapply(state_vec, function (i) chol(x$hreg$Sigmak[, , i]))

  # Get the Z vector if it is not provided
  if (is.null(Z_init)) {
    Z_init <- tail(embed(x$init.model$Y, x$p), 1)
  }

  Bk_cpp <- matrix(x$hreg$Bk, dim(x$hreg$Bk)[1])
  sigmaU_cpp <- matrix(NA, nrow(sigmaU[[1]]), nrow(sigmaU[[1]]) * length(sigmaU))
  for (i in seq_along(sigmaU)) {
    sigmaU_cpp[, ((i - 1) * nrow(sigmaU[[1]]) + 1):(i * nrow(sigmaU[[1]]))] <- sigmaU[[i]]
  }

  forecasts <- predict_cpp(samples, x$Q, Bk_cpp, sigmaU_cpp, tail(x$fp, 1), h, x$m, x$p, Z_init)
  forecasts <- array(forecasts, dim = c(h, x$m, samples))

  return(forecasts)
}

