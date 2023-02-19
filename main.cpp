#include <iostream>
#include <vector>

#include "mnist_pre_process.h"
#include "random_forest.h"

#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000
#define FEATURE_SIZE 784
#define NUM_CLASSES 10

int main(int argc, const char* argv[]) {
  // 定义数据容器
  std::vector<std::vector<float>> train_set(TRAIN_SET_SIZE,
                                            std::vector<float>(FEATURE_SIZE));
  std::vector<std::vector<float>> test_set(TEST_SET_SIZE,
                                           std::vector<float>(FEATURE_SIZE));
  std::vector<float> train_labels(TRAIN_SET_SIZE);
  std::vector<float> test_labels(TEST_SET_SIZE);
  
  // 读取数据
  ReadData(train_set, train_labels, "./train-images.idx3-ubyte",
           "./train-labels.idx1-ubyte");
  ReadData(test_set, test_labels, "./t10k-images.idx3-ubyte",
           "./t10k-labels.idx1-ubyte");

  // 创建随机森林
  RandomForest random_forest(20, 10, 10, 0);

  // 开始训练
  random_forest.Train(&train_set, &train_labels, TRAIN_SET_SIZE, FEATURE_SIZE,
                      NUM_CLASSES, "sqrt");

  // 预测结果
  std::vector<float> resopnses(TEST_SET_SIZE);
  random_forest.Predict(&test_set, TEST_SET_SIZE, &resopnses);
  float error_rate = 0;
  for (int i = 0; i < TEST_SET_SIZE; ++i) {
    if (resopnses[i] != test_labels[i]) {
      error_rate += 1.0f;
    }
  }
  error_rate /= TEST_SET_SIZE;
  std::cout << "测试集错误率为：" << error_rate << std::endl;
  return 0;
};
