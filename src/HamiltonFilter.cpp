#include <RcppEigen.h>
#include <algorithm>

// [[Rcpp::depends(RcppEigen)]]

//' Hamilton filter
//'
//' @export
// [[Rcpp::export]]
Rcpp::List hamiltonFilter(
    int bigt,
    int m,
    int p,
    int h,
    Eigen::Map<Eigen::MatrixXd> e,
    Eigen::Map<Eigen::MatrixXd> sig2,
    Eigen::Map<Eigen::MatrixXd> Qhat
) {
  // Loop over regimes (1 to h) to calculate univariate Normal
  // or multivariate Normal density given parameter values
  Eigen::MatrixXd ylik = Eigen::MatrixXd::Zero(bigt - p, h);
  Eigen::MatrixXd ypwlik = Eigen::MatrixXd::Zero(bigt - p, h * h);
  for (int iterh = 0; iterh < h; iterh++) {
    double detsig2 = std::abs(sig2.middleCols(iterh * m, m).determinant());
    Eigen::MatrixXd invsig2 = sig2.middleCols(iterh * m, m).inverse();
    for (int itert = 0; itert < bigt - p; itert++) {
      Eigen::RowVectorXd tmpfit = e.middleCols(iterh * m, m).row(itert);
      Eigen::MatrixXd matmultwo = tmpfit * invsig2 * tmpfit.transpose();
      ylik(itert, iterh) = std::max(std::exp(-(m / 2.0) * std::log(2.0 * M_PI) - (0.5 * std::log(detsig2)) - (0.5 * matmultwo(0, 0))), std::numeric_limits<double>::epsilon());
    }
  }

  for (int iterh1 = 0; iterh1 < h; iterh1++) {
    for (int iterh2 = 0; iterh2 < h; iterh2++) {
      ypwlik.col(iterh1 + iterh2 * h) = Qhat(iterh1, iterh2) * ylik.col(iterh2);
    }
  }

  Eigen::MatrixXd ssAmat = Eigen::MatrixXd::Constant(h + 1, h, 1.0);
  Eigen::VectorXd ssEvec = Eigen::VectorXd::Zero(h + 1);
  ssEvec(h) = 1.0;
  ssAmat.topRows(h) = Eigen::MatrixXd::Identity(h, h) - Qhat.transpose();

  Eigen::MatrixXd Invmatss = (ssAmat.transpose() * ssAmat).inverse();

  Eigen::VectorXd pSt1_t1 = Invmatss * ssAmat.transpose() * ssEvec;

  double f = 0.0;
  Eigen::MatrixXd pytSt1St_t1_itert = Eigen::MatrixXd::Zero(h, h);
  Eigen::MatrixXd filtprSt1St = Eigen::MatrixXd::Zero(bigt - p, h * h);

  for (int its = 0; its < bigt - p; its++) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < h; j++) {
        pytSt1St_t1_itert(i, j) = pSt1_t1(i) * ypwlik(its, j * h + i);
      }
    }
    double filt_llfval = pytSt1St_t1_itert.sum();
    f += std::log(filt_llfval);

    pytSt1St_t1_itert = pytSt1St_t1_itert / filt_llfval;

    filtprSt1St.row(its) = pytSt1St_t1_itert.reshaped();
    pSt1_t1 = pytSt1St_t1_itert.colwise().sum();
  }

  // integrate over St-1
  Eigen::MatrixXd filtprSt = Eigen::MatrixXd::Zero(bigt - p, h);
  for (int i = 0; i < h; i++) {
    filtprSt += filtprSt1St.middleCols(i * h, h);
  }

  return Rcpp::List::create(Rcpp::Named("filtprSt") = filtprSt, Rcpp::Named("f") = f, Rcpp::Named("e") = e);
}