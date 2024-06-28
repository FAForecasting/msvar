#include "EigenReadWrite.h"

void writeMatrixToFile(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << matrix.rows() << " " << matrix.cols() << "\n";
        file << matrix << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}

void writeVectorToFile(const Eigen::VectorXd& vector, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << vector.size() << "\n";
        file << vector.transpose() << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}

Eigen::MatrixXd readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    int rows, cols;
    file >> rows >> cols;
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    file.close();
    return matrix;
}

Eigen::VectorXd readVectorFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return Eigen::VectorXd();
    }

    int size;
    file >> size;
    Eigen::VectorXd vector(size);
    for (int i = 0; i < size; ++i) {
        file >> vector(i);
    }
    file.close();
    return vector;
}