#ifndef MATRIXBATCH_H
#define MATRIXBATCH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mkl.h>
#include <omp.h>
#include <cmath>

// #include "saving.hpp"

#define Complex MKL_Complex16
#define Float double
#define GEEV LAPACKE_zgeev
#define GEQRF LAPACKE_zgeqrf
#define UNGQR LAPACKE_zungqr

inline void print_type_info() {
    std::cout << "Complex type: " << typeid(Complex).name() << std::endl;
    std::cout << "Float type: " << typeid(Float).name() << std::endl;
    std::cout << "GEEV function: " << 
        #ifndef USE_SINGLE_PRECISION
            "LAPACKE_zgeev"
        #else
            "LAPACKE_cgeev"
        #endif
        << std::endl;
}

inline std::ostream& operator<<(std::ostream &os, const Complex c) {
  os << std::fixed << std::setprecision(4);
  if (c.real >= 0) os << " " <<  c.real;
  else             os << "-" << -c.real;
  if (c.imag >= 0) os << " + " <<  c.imag << "j";
  else             os << " - " << -c.imag << "j";
  return os;
}

struct MatrixBatch {
  unsigned int rows, cols, amount;
  Complex *data, *eigenBuffer;

  MatrixBatch(unsigned int rows, unsigned int cols, unsigned int amount) : rows(rows), cols(cols), amount(amount), data(nullptr), eigenBuffer(nullptr)  {
    data = (Complex*) malloc(amount * rows * cols * sizeof(Complex));
  }

  void free() {
    std::free(data);
    if (eigenBuffer!= nullptr) {
      std::free(eigenBuffer);
    }
  }
  void print(int k) {
    if (k >= amount) {
      std::cout << "there is no " << k << "-th matrix" << std::endl;
      return;
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        std::cout << std::fixed << std::setprecision(8) << (data + k * rows * cols)[j + i * cols] << "    ";
      }
      std::cout << "\n";
    }
  }
  bool isHermitian(int k) {
    if (rows != cols) return false;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if ((data + k * rows * cols)[j + i * cols].real !=  (data + k * rows * cols)[i + j * cols].real) return false;
        if ((data + k * rows * cols)[j + i * cols].imag != -(data + k * rows * cols)[i + j * cols].imag) return false;
      }
    }
    return true;
  }
  Complex trace(int k) {
    if (rows != cols) std::runtime_error("Trace is not defined for non-square matrices");
    Complex tr;
    tr.real = 0.0;
    tr.imag = 0.0;
    for (int i = 0; i < rows; i++) {
      tr.real += (data + k * rows * cols)[i + i * cols].real;
      tr.imag += (data + k * rows * cols)[i + i * cols].imag;
    }
    return tr;
  }
  void eigenvalues(int k) {
    if (rows != cols) std::runtime_error("Eigenvalues are not defined for non-square matrices");
    if (eigenBuffer == nullptr) {
      eigenBuffer = (Complex*) malloc(rows * sizeof(Complex));
    }

    Complex *A = (Complex*)malloc(rows*cols * sizeof(Complex));

    for (int i = 0; i < rows * cols; i++) {
      A[i].real = (data + k * rows * cols)[i].real;
      A[i].imag = (data + k * rows * cols)[i].imag;
    }

    GEEV(LAPACK_ROW_MAJOR, 'N', 'N', rows, A, cols, eigenBuffer, NULL, 1, NULL, 1);

    std::free(A);
  }
  bool isPositive(int k) {
    eigenvalues(k);

    float eps = 0.000001;
    for (int i = 0; i < rows; i++) {
      if ((eigenBuffer[i].real < 0 && abs(eigenBuffer[i].real) > eps) || std::abs(eigenBuffer[i].imag) > eps) {
        return false;
      }
    }
    return true;
  }
  static void kronecker(MatrixBatch A, MatrixBatch B, MatrixBatch C) {
    if (C.rows != A.rows * B.rows) std::runtime_error("Kronecker product: number of rows don't match");
    if (C.cols != A.cols * B.cols) std::runtime_error("Kronecker product: number of cols don't match");
    if (C.amount != A.amount * B.amount) std::runtime_error("Kronecker product: amount of matrices don't match");

    for (int k = 0; k < C.amount; k++) {
      for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
          (C.data + k*C.rows*C.cols)[j + i * C.cols].real
            = (A.data + k*A.rows*A.cols)[j / B.cols + i / B.rows * A.rows].real
            * (B.data + k*B.rows*B.cols)[j % B.cols + i % B.rows * B.rows].real 
            - (A.data + k*A.rows*A.cols)[j / B.cols + i / B.rows * A.rows].imag
            * (B.data + k*B.rows*B.cols)[j % B.cols + i % B.rows * B.rows].imag;
          (C.data + k*C.rows*C.cols)[j + i * C.cols].imag
            = (A.data + k*A.rows*A.cols)[j / B.cols + i / B.rows * A.rows].real
            * (B.data + k*B.rows*B.cols)[j % B.cols + i % B.rows * B.rows].imag
            + (A.data + k*A.rows*A.cols)[j / B.cols + i / B.rows * A.rows].imag
            * (B.data + k*B.rows*B.cols)[j % B.cols + i % B.rows * B.rows].real;
        }
      }
    }
  }
  static void partialTransposeArray(Complex* A, Complex*C, unsigned int dim1, unsigned int dim2) {
    unsigned int rows = dim1 * dim2;
    unsigned int cols = dim1 * dim2;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim1; j++) {
        for (int m = 0; m < dim2; m++) {
          for (int n = 0; n < dim2; n++) {
            C[j * dim2 + m + (i * dim2 + n) * cols].real += 
              A[j * dim2 + n + (i * dim2 + m) * cols].real;
            C[j * dim2 + m + (i * dim2 + n) * cols].imag += 
              A[j * dim2 + n + (i * dim2 + m) * cols].imag;
          }
        }
      }
    }
  }
  static void partialTransposeBatch(MatrixBatch A, MatrixBatch C, unsigned int dim1, unsigned int dim2) {
    if (C.rows != A.cols) std::runtime_error("Transpose: number of C rows don't match A cols");
    if (C.cols != A.rows) std::runtime_error("Transpose: number of A cols don't match C rows");
    if (C.amount != A.amount) std::runtime_error("Transpose: amount of matrices don't match");
  
    for (int i = 0; i < C.amount * C.rows * C.cols; i++) { C.data[i].real = 0.0; C.data[i].imag = 0.0; }
    for (int k = 0; k < A.amount; k++) {
      partialTransposeArray(A.data + k * A.rows * A.cols, C.data + k * A.rows * A.cols, dim1, dim2);
    }
  }
  static void batchInnerProduct(MatrixBatch A, MatrixBatch B, MatrixBatch C) {
    if (C.amount) std::runtime_error("Inner product: results are in only 1 matrix");
    if (C.rows != A.amount) std::runtime_error("Inner product: number of rows in C don't match amount of A");
    if (C.cols != B.amount) std::runtime_error("Inner product: number of cols in C don't match amount of B");
    if (A.rows != B.rows) std::runtime_error("Inner product: number of rows in A and B don't match");
    if (A.cols != B.cols) std::runtime_error("Inner product: number of cols in A and B don't match");

    #pragma omp parallel for
    for (int s = 0; s < A.amount; s++) {
      for (int w = 0; w < B.amount; w++) {
        C.data[w + s * C.cols].real = 0.0;
        C.data[w + s * C.cols].imag = 0.0;
        for (int i = 0; i < A.rows * A.cols; i++) {
          C.data[w + s * C.cols].real +=         (A.data + s * A.rows * A.cols)[i].real
                                               * (B.data + w * B.rows * B.cols)[i].real
                                               + (A.data + s * A.rows * A.cols)[i].imag
                                               * (B.data + w * B.rows * B.cols)[i].imag;

          C.data[w + s * C.cols].imag +=         (A.data + s * A.rows * A.cols)[i].real
                                               * (B.data + w * B.rows * B.cols)[i].imag
                                               - (A.data + s * A.rows * A.cols)[i].imag
                                               * (B.data + w * B.rows * B.cols)[i].real;
        }
      }
    }
  }
};

struct SimulationParameters {
  unsigned int dim1;
  unsigned int dim2;
  unsigned int seedS;
  unsigned int seedW;
  unsigned int amountS;
  unsigned int amountW;
  unsigned int NSi;
  unsigned int NSf;
  unsigned int NWi;
  unsigned int NWf;
  std::string to_string() {
    std::ostringstream filename;
    filename << "dim1=" << dim1 << "_dim2=" << dim2;
    filename << "_seedS=" << seedS << "_seedW=" << seedW;
    filename << "_NSi=" << NSi << "_NSf=" << NSf;
    filename << "_NWi=" << NWi << "_NWf=" << NWf;
    filename << ".bin";
    return filename.str();
  }
  static SimulationParameters from_string(const std::string& filename) {
    SimulationParameters params;
    std::string temp = filename;

    // Remove ".bin" extension if present
    if (temp.length() > 4 && temp.substr(temp.length() - 4) == ".bin") {
      temp = temp.substr(0, temp.length() - 4);
    }

    std::istringstream iss(temp);
    std::string token;

    while (std::getline(iss, token, '_')) {
      std::cout << token << std::endl;

      std::string key, value;
      auto pos = token.find('=');
      if (pos != std::string::npos) {
        key = token.substr(0, pos);
        value = token.substr(pos + 1);

        if (key == "dim1") params.dim1 = std::stoull(value);
        else if (key == "dim2") params.dim2 = std::stoull(value);
        else if (key == "seedS") params.seedS = std::stoul(value);
        else if (key == "seedW") params.seedW = std::stoul(value);
        else if (key == "NSi") params.NSi = std::stoul(value);
        else if (key == "NSf") params.NSf = std::stoul(value);
        else if (key == "NWi") params.NWi = std::stoul(value);
        else if (key == "NWf") params.NWf = std::stoul(value);
      }
    }
    return params;
  }
};

struct Histogram2D {
  unsigned int nhists, nbins;
  Float min, max, delta;
  size_t *counts;

  Histogram2D(unsigned int nhists, unsigned int nbins, Float min, Float max) : nhists(nhists), nbins(nbins), min(min), max(max) {
    counts = (size_t*)malloc(nhists * 2 * nbins * sizeof(size_t));
    for (int i = 0; i < nhists * 2 * nbins; i++) counts[i] = 0;
    delta = (max - min) / nbins;
  }
  void free() {
    std::free(counts);
  }
  void addCount(unsigned int hist, double val, Float ent) {
    int binIndex0 = ent > 0.5 ? 1 : 0;
    int binIndex1 = static_cast<int>(std::floor((val - min) / delta));
    counts[hist * 2 * nbins + binIndex0 * nbins + binIndex1]++;
  }
  void save(SimulationParameters param, int reps) {
    std::ostringstream filename;
    filename << "hist_";
    filename << "dim1=" << param.dim1 << "_";
    filename << "dim2=" << param.dim2 << "_";
    filename << "min=" << min << "_";
    filename << "max=" << max << "_";
    filename << "nbins=" << nbins << "_";
    filename << "seedS=" << param.seedS << "_";
    filename << "seedW=" << param.seedW << "_";
    filename << "amountS=" << param.amountS * reps << "_";
    filename << "amountW=" << param.amountW;

    std::ofstream ofile;
    ofile.open(filename.str().c_str());

    for (int h = 0; h < nhists; h++) { // fixing a witnesss
      for (int b = 0; b < nbins; b++) { // not entangled
        ofile << counts[h * nbins * 2 + b];
        if (b != nbins - 1) ofile << ", ";
      }
      ofile << "\n";
      for (int b = 0; b < nbins; b++) { // entangled
        ofile << counts[h * nbins * 2 + nbins + b];
        if (b != nbins - 1) ofile << ", ";
      }
      ofile << "\n";
    }
    ofile.close();

  }
};

#endif
