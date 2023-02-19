#ifndef MNISTPREPROCESS_H_
#define MNISTPREPROCESS_H_

#include <vector>

inline void RevertInt(int& x) {
  x = ((x & 0x000000ff) << 24) | ((x & 0x0000ff00) << 8) |
      ((x & 0x00ff0000) >> 8) | ((x & 0xff000000) >> 24);
};
void ReadData(std::vector<std::vector<float>>& dataset,
              std::vector<float>& labels, const char* data_path,
              const char* label_path);
#endif  // MNISTPREPROCESS_H
