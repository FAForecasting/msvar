#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' predict_internal
//'
// [[Rcpp::export]]
Eigen::MatrixXd predict_cpp(
    int samples,
    Eigen::Map<Eigen::MatrixXd> Q,
    Eigen::Map<Eigen::MatrixXd> Bk,
    Eigen::Map<Eigen::MatrixXd> sigmaU,
    Eigen::Map<Eigen::VectorXd> fp_last,
    int h,
    int m,
    int p,
    Eigen::Map<Eigen::VectorXd> Z_init
) {
    Eigen::MatrixXd fcsts = Eigen::MatrixXd(h, m * samples);
    Eigen::MatrixXd fcst = Eigen::MatrixXd(h, m);
    Eigen::VectorXi path = Eigen::VectorXi(h);
    Eigen::VectorXd Z = Eigen::VectorXd(Z_init.size() + 1);
    Eigen::MatrixXd A = Eigen::MatrixXd(m * p + 1, m);
    Eigen::VectorXd tmp = Eigen::VectorXd(m * (p - 1));
    for (int sample = 0; sample < samples; sample++) {
        // Simulate the path of states
        path(0) = Rcpp::sample((int) Q.cols(), 1, true, Rcpp::wrap(fp_last), false)[0];
        for (int i = 1; i < h; i++) {
            path(i) = Rcpp::sample((int) Q.cols(), 1, true, Rcpp::wrap(Q.row(path(i - 1))), false)[0];
        }
        // Get the initial Z vector
        Z << Z_init, 1.0;
        for (int i = 0; i < h; i++) {
            // Get the coefficient matrix to use
            A = Bk.middleCols(path[i] * m, m);
            // Calculate prediction and add error sample
            fcst.row(i) = A.transpose() * Z + sigmaU.middleCols(path[i] * m, m) * Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Rcpp::rnorm(m));

            // Update Z with the prediction
            tmp = Z.segment(0, m * (p - 1));
            Z.segment(m, m * (p - 1)) = tmp;
            Z.segment(0, m) = fcst.row(i);
        }
        fcsts.middleCols(sample * m, m) = fcst;
    }

    return fcsts;
}