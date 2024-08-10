#include <mkl.h>
#include <mkl_lapacke.h>
#include <limits.h>
#include <omp.h>
#include <functional>

#include "matrixbatch.h"
// #include "saving.hpp"

using namespace std;

/* Using a few methods from Intel MKL
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/geqrf.html
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/ungqr.html
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/geev.html
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/vs-rng-usage-model-intel-onemkl-rng-usage-model.html
 * */ 


struct BipartiteStatesBatch : public MatrixBatch {
  VSLStreamStatePtr rdStream;  

  BipartiteStatesBatch(unsigned int dim1, unsigned int dim2, unsigned int amount, unsigned int seed) : MatrixBatch(dim1*dim2, dim1*dim2, amount) {
    vslNewStream(&rdStream, VSL_BRNG_MT19937, seed);
    randomize();
  }
  void randomize() {
    Float mean = 0.0, sigma = 1.0; // gaussian distribution parameters
    
    int unitaryRows = rows*cols; // dimension of a purification of the state
    int unitaryCols = rows*cols;
    
    Complex* A = (Complex*) malloc(unitaryRows * unitaryCols * sizeof(Complex));
    Complex* tau = (Complex*) malloc(unitaryCols * sizeof(Complex));

    for (int k = 0; k < amount; k++) {
      // randomize A
      for (int i = 0; i < unitaryRows * unitaryCols; i++) {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rdStream, 1, &A[i].real, mean, sigma);
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rdStream, 1, &A[i].imag, mean, sigma);
      }
      // QR decompose A, get only first column of Q
      GEQRF (LAPACK_ROW_MAJOR, unitaryRows, unitaryRows, A, unitaryCols, tau);
      UNGQR (LAPACK_ROW_MAJOR, unitaryRows, unitaryCols, 1, A, unitaryCols, tau); // 1 because we only need the first column of U

      // store first column of Q in tau
      for (int i = 0; i < unitaryRows; i++) {
        tau[i].real = A[i * unitaryCols].real;
        tau[i].imag = A[i * unitaryCols].imag; 
      }

      // partial trace of U|0>
      for (int i = 0; i < rows; i++) {
        for (int m = 0; m < cols; m++) {
          (data + k * rows * cols)[m + i * cols].real = 0.0;
          (data + k * rows * cols)[m + i * cols].imag = 0.0;
          for (int p = 0; p < rows; p++) {
            (data + k * rows * cols)[m + i * cols].real += 
              tau[i * rows + p].real * tau[m * rows + p].real
            + tau[i * rows + p].imag * tau[m * rows + p].imag;
            (data + k * rows * cols)[m + i * cols].imag += 
              tau[i * rows + p].real * tau[m * rows + p].imag
            - tau[i * rows + p].imag * tau[m * rows + p].real;
          }
        }
      }
    }

    std::free(A);
    std::free(tau);
  }
  void free() {
    MatrixBatch::free();
    vslDeleteStream(&rdStream);
  }
};

struct WitnessPartialTransposeBatch : public MatrixBatch {
  VSLStreamStatePtr rdStream;  
  unsigned int dim1, dim2;

  WitnessPartialTransposeBatch(unsigned int dim1, unsigned int dim2, unsigned int amount, unsigned int seed) : MatrixBatch(dim1*dim2, dim1*dim2, amount) {
    this->dim1 = dim1;
    this->dim2 = dim2;
    vslNewStream(&rdStream, VSL_BRNG_MT19937, seed);
    randomize();
  }
  void randomize() {
    Float mean = 0.0, sigma = 1.0; // gaussian distribution parameters

    int unitaryRows = rows; // dimension of a purification of the state
    int unitaryCols = cols;
    
    Complex* A = (Complex*) malloc(unitaryRows * unitaryCols * sizeof(Complex));
    Complex* tau = (Complex*) malloc(unitaryCols * sizeof(Complex));

    for (int k = 0; k < amount; k++) {
      // randomize A
      for (int i = 0; i < unitaryRows * unitaryCols; i++) {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rdStream, 1, &A[i].real, mean, sigma);
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rdStream, 1, &A[i].imag, mean, sigma);
      }
      // QR decompose A, get only first column of Q
      GEQRF (LAPACK_ROW_MAJOR, unitaryRows, unitaryRows, A, unitaryCols, tau);
      UNGQR (LAPACK_ROW_MAJOR, unitaryRows, unitaryCols, 1, A, unitaryCols, tau); // 1 because we only need the first column of U

      // partial Transpose of U|0><0|U^dagger over second system
      for (int i = 0; i < unitaryRows; i++) { // reutilizing tau since dimensiosn fit
        tau[i].real = A[i * unitaryCols].real;
        tau[i].imag = A[i * unitaryCols].imag; 
      }

      // Partial transpose directly from the wave function
      for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
          for (int m = 0; m < dim1; m++) {
            for (int n = 0; n < dim2; n++) {
              (data + k *  rows * cols)[m * dim2 + n +(i * dim2 + j) * cols].real
                += tau[i * dim2 + n].real * tau[m * dim2 + j].real
                 + tau[i * dim2 + n].imag * tau[m * dim2 + j].imag;
              (data + k *  rows * cols)[m * dim2 + n +(i * dim2 + j) * cols].imag
                += tau[i * dim2 + n].real * tau[m * dim2 + j].imag
                 - tau[i * dim2 + n].imag * tau[m * dim2 + j].real;
            }
          }
        }
      }

      // Normalize to unit trace
      Float tr = 0.0;
      for (int i = 0; i < rows; i++) {
        tr += (data + k * rows * cols)[i + i * cols].real;
      }
      for (int i = 0; i < rows * cols; i++) {
        (data + k * rows * cols)[i].real /= tr;
        (data + k * rows * cols)[i].imag /= tr;
      }
    }

    std::free(A);
    std::free(tau);
  }

  void free() {
    MatrixBatch::free();
    vslDeleteStream(&rdStream);
  }
};

void sep() {
  std::cout << "------------\n";
}

struct EntanglementLabels {
  Float* data = nullptr;
  unsigned int length;
  std::function<Float(Complex*)> f;
  EntanglementLabels(std::function<Float(Complex*)> f, unsigned int amountS) {
    this->f = f;
    data = (Float*)malloc(amountS * sizeof(Float));
    length = amountS;
  }
  void genLabels(MatrixBatch S) {
    for (int k = 0; k < length; k++) {
      data[k] = f(S.data + k * S.rows * S.cols);
    }
  }
  void free() {
    if (data != nullptr) {
      std::free(data);
    }
  }
};

Float PPT_2x2(Complex* Si) {
    Complex *eigenBuffer = (Complex*) malloc(4 * sizeof(Complex));
    Complex *Sit = (Complex*)malloc(4*4 * sizeof(Complex));

    for (int i = 0; i < 4*4; i++) { Sit[i].real = 0.0; Sit[i].imag = 0.0; }
    MatrixBatch::partialTransposeArray(Si, Sit, 2, 2);

    GEEV(LAPACK_ROW_MAJOR, 'N', 'N', 4, Sit, 4, eigenBuffer, NULL, 1, NULL, 1);

    std::free(Sit);

    float eps = 0.000001;
    for (int i = 0; i < 4; i++) {
      if ((eigenBuffer[i].real < 0 && abs(eigenBuffer[i].real) > eps) || std::abs(eigenBuffer[i].imag) > eps) {
        std::free(eigenBuffer);
        return 1.0;
      }
    }
    std::free(eigenBuffer);
    return 0.0;
}

Float PPT_2x3(Complex* Si) {
    Complex *eigenBuffer = (Complex*) malloc(2*3 * sizeof(Complex));
    Complex *Sit = (Complex*)malloc(6*6 * sizeof(Complex));

    for (int i = 0; i < 6*6; i++) { Sit[i].real = 0.0; Sit[i].imag = 0.0; }
    MatrixBatch::partialTransposeArray(Si, Sit, 2, 3);

    GEEV(LAPACK_ROW_MAJOR, 'N', 'N', 6, Sit, 6, eigenBuffer, NULL, 1, NULL, 1);

    std::free(Sit);

    for (int i = 0; i < 6; i++) {
      if (eigenBuffer[i].real < 0){
        std::free(eigenBuffer);
        return 1.0;
      }
    }
    std::free(eigenBuffer);
    return 0.0; 
}



void fixingEntanglementBug();
void testHist();

int main(int argc, char **argv) {
  testHist();
}

void fixingEntanglementBug () {
  SimulationParameters P;
  P.dim1 = 2;
  P.dim2 = 3;
  P.seedS = 1;
  P.seedW = 2;
  P.amountS = 1000;
  P.amountW = 100;
  P.NSi = 0;
  P.NSf = P.amountS - 1;
  P.NWi = 0;
  P.NWf = P.amountW - 1;

  BipartiteStatesBatch S(P.dim1, P.dim2, P.amountS, P.seedS);
  MatrixBatch St(P.dim1 * P.dim2, P.dim1 * P.dim2, P.amountS);
  EntanglementLabels E(&PPT_2x3, P.amountS);
  E.genLabels(S);
  
  MatrixBatch::partialTransposeBatch(S, St, P.dim1, P.dim2);

  cout << S.amount << " " << St.amount << " " << P.amountS << endl;

  unsigned int entCount = 0;
  unsigned int entCountPPT = 0;
  for (int i = 0; i < P.amountS; i++) {
//     S.print(i);
//     sep();
//     St.print(i);
    cout << S.isPositive(i) << " | " << !St.isPositive(i) << " | " << E.data[i] << endl;
//     cout << S.isPositive(i) << " | " << !St.isPositive(i) << " | " << PPT_2x2(S.data + i * S.rows * S.cols) << endl;
//     sep();
//     sep();
    if (!St.isPositive(i)) entCount++;
    if (E.data[i] == 1.0) entCountPPT++;
  }

  cout << entCount / (double) P.amountS << endl;
  cout << entCountPPT / (double) P.amountS << endl;
  cout << (entCount == entCountPPT) << endl;
}

void testHist() {
  double start, end;
  double total_start, total_end;

  SimulationParameters param;
  param.dim1 = 2;
  param.dim2 = 3;
  param.seedS = 1;
  param.seedW = 2;
  param.amountS = 100;
  param.amountW = 100000;
  param.NSi = 0;
  param.NSf = param.amountS - 1;
  param.NWi = 0;
  param.NWf = param.amountW - 1;

  start = dsecnd();
  WitnessPartialTransposeBatch W(param.dim1, param.dim2, param.amountW, param.seedW);
  end = dsecnd();
  BipartiteStatesBatch S(param.dim1, param.dim2, param.amountS, param.seedS);
  EntanglementLabels E(&PPT_2x3, param.amountS);
  MatrixBatch trWrho(param.amountW, param.amountS, 1);

  cout << end - start << " seconds to generate " << param.amountW << " witnesses " << endl;

  unsigned int nbins = 128;
  Float min = -1.0, max = 1.0;

  Histogram2D hist128(param.amountW, nbins, min, max);
  
  int reps = 1000;
  unsigned int ent_count = 0;

  total_start = dsecnd();
  for (int i = 0; i < reps; i++) {
    param.NSi = param.amountS * i;
    param.NSf = param.amountS * (i + 1) - 1;
    cout << "Progress: " << param.NSi / (double) (param.amountS * reps) * 100 << "%\r";
    cout.flush();

    S.randomize();
    E.genLabels(S);
    MatrixBatch::batchInnerProduct(W, S, trWrho);

    for (int i = 0; i < param.amountS; i++) {
      if (E.data[i] > 0.0) ent_count++;
    }

  for (int w = 0; w < param.amountW; w++) {
    for (int s = 0; s < param.amountS; s++) {
        hist128.addCount(w, trWrho.data[s + w * param.amountS].real, E.data[s]);
      }
    }
  }
  cout << endl;
  total_end = dsecnd();
  S.free();
  E.free();
  W.free();
  trWrho.free();

  hist128.save(param, reps);
  hist128.free();

  cout << total_end - total_start << " seconds to calculate ";
  cout << param.amountS << "x" << param.amountW << "x" << reps << " inner products " << endl;
  cout << ent_count << endl;
}
