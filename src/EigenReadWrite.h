#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

void writeMatrixToFile(const Eigen::MatrixXd& matrix, const std::string& filename);
void writeVectorToFile(const Eigen::VectorXd& vector, const std::string& filename);

Eigen::MatrixXd readMatrixFromFile(const std::string& filename);
Eigen::VectorXd readVectorFromFile(const std::string& filename);