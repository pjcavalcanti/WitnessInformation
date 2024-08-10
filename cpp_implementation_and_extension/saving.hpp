#ifndef SAVING_H
#define SAVING_H

#include <fstream>
#include <sstream>
#include <string>

#include "matrixbatch.h"


// void save(EntanglementLabels E, MatrixBatch trWrho, SaveParameters param) {
//   ofstream ofile(param.to_string().c_str(), ios::binary | ios::out);
//   ofile.write((char*)&param, sizeof(SaveParameters));
//   for (int s = 0; s < param.amountS; s++) {
//     ofile.write((char*)(E. data + s), sizeof(Float));
//     ofile.write((char*)(trWrho.data + s * param.amountW), param.amountW * sizeof(Float));
//   }
//   ofile.close();
// }

void save(Float* entanglement_label_data, Complex * trWrho_data, SimulationParameters param) {
  std::ofstream ofile(param.to_string().c_str(), std::ios::binary | std::ios::out);

  ofile.write((char*)&param, sizeof(SimulationParameters));
  for (int s = 0; s < param.amountS; s++) {
    ofile.write((char*)(entanglement_label_data + s), sizeof(Float));
    for (int w = 0; w < param.amountW; w++) {
      ofile.write((char*)&(trWrho_data + w + s * param.amountW)->real, sizeof(Float));
    }
//     ofile.write((char*)(trWrho_data + s * param.amountW), param.amountW * sizeof(Complex));
  }
  ofile.close();
}

// void load(EntanglementLabels E, MatrixBatch trWrho, std::string filename) {
//   ifstream infile;
//   SaveParameters param = SaveParameters::from_string(filename);
// 
//   infile.open("test", ios::binary | ios::in);

  
//   double * y = (double*) malloc(length * sizeof(double));
//   infile.read((char*)y, length * sizeof(double));
//   infile.close();
// 
//   for (int i = 0; i < length; i++) {
//     cout << y[i] << endl;
//   }
// 
// }

#endif
