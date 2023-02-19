#include "random_forest.h"

#include "unistd.h"

#include <CL/sycl.hpp>

#include <iostream>
#include <thread>

RandomForest::RandomForest(int num_trees, int max_depth, int min_leaf_sample,
                           float min_info_gain) {
  num_trees_ = num_trees;
  max_depth_ = max_depth;
  min_leaf_num_samples_ = min_leaf_sample;
  min_info_gain_ = min_info_gain;
  train_samples_ = NULL;
  std::cout << "树的数量：" << num_trees_ << std::endl;
  std::cout << "树的最大深度：" << max_depth_ << std::endl;
  std::cout << "叶子上最少的样本数：" << min_leaf_num_samples_ << std::endl;
  std::cout << "分裂时最少的基尼系数增益：" << min_info_gain_ << std::endl;

  trees_.resize(num_trees_);
  trees_.assign(num_trees_, NULL);
}

RandomForest::~RandomForest() {
  for (int i = 0; i < num_trees_; ++i) {
    if (trees_[i] != NULL) {
      delete trees_[i];
      trees_[i] = NULL;
    }
  }
  if (train_samples_ != NULL) {
    delete train_samples_;
    train_samples_ = NULL;
  }
}

void RandomForest::Train(std::vector<std::vector<float>>* train_set,
                         std::vector<float>* labels, int num_samples,
                         int num_features, int num_classes, std::string mode) {
  int num_train_features_per_node = 0;
  if (mode == "sqrt") {
    num_train_features_per_node =
        static_cast<int>(sqrt(static_cast<double>(num_features)));
  } else if (mode == "log2") {
    num_train_features_per_node =
        static_cast<int>(log2(static_cast<double>(num_features)));
  }
  if (num_trees_ < 1) {
    std::cout << "树的数量必须大于 0 ！" << std::endl
              << "训练失败" << std::endl;
    return;
  }
  if (max_depth_ < 1) {
    std::cout << "树的最大深度必须大于 0 ！" << std::endl
              << "训练失败" << std::endl;
    return;
  }
  if (min_leaf_num_samples_ < 2) {
    std::cout << "叶子上的最少样本数必须大于 1 ！" << std::endl
              << "训练失败" << std::endl;
    return;
  }
  num_train_samples_ = num_samples;
  num_features_ = num_features;
  num_classes_ = num_classes;
  num_train_features_per_node_ = num_train_features_per_node;
  // 初始化每一棵树
  std::vector<sycl::queue*> qs(num_trees_);
  for (int i = 0; i < num_trees_; ++i) {
    qs[i] = new sycl::queue(sycl::cpu_selector_v);
    trees_[i] = new Tree(max_depth_, num_train_features_per_node_,
                         min_leaf_num_samples_, min_info_gain_, qs[i]);
  }
  // 训练数据集对象
  train_samples_ = new SampleSet(train_set, labels, num_classes_,
                                 num_train_samples_, num_features_);
  srand(static_cast<unsigned int>(time(NULL)));
  // 并行训练森林中的每一棵树
  std::vector<std::thread*> threads(num_trees_);
  for (int i = 0; i < num_trees_; i++) {
    threads[i] = new std::thread([=]() {
      printf("开始训练第 %d 棵树...\n", i);
      // 随机采样样本
      SampleSet* samples = new SampleSet(train_samples_);
      samples->RandomSelectSample(num_train_samples_, num_train_samples_);
      trees_[i]->Train(samples);
      delete samples;
    });
    // threads[i]->join();
  }
  for (int i = 0; i < num_trees_; i++) {
    if (threads[i]->joinable()) {
      threads[i]->join();
    }
    delete threads[i];
  }
}

void RandomForest::PredictOne(std::vector<float>* data, float* response) {
  // 获取每棵树的结果
  std::vector<float> result(num_classes_);
  int i = 0;
  for (i = 0; i < num_classes_; ++i) {
    result[i] = 0;
  }
  for (i = 0; i < num_trees_; ++i) {
    Result r;
    r.label = 0;
    r.prob = 0;  // 结果
    r = trees_[i]->Predict(data);
    result[static_cast<int>(r.label)] += r.prob;
  }
  float max_prob_label = 0;
  float max_prob = result[0];
  for (i = 1; i < num_classes_; ++i) {
    if (result[i] > max_prob) {
      max_prob_label = i;
      max_prob = result[i];
    }
  }
  *response = max_prob_label;
}

void RandomForest::Predict(std::vector<std::vector<float>>* test_set,
                           int num_samples, std::vector<float>* responses) {
  // 获取每棵树对每个样本的预测
  std::vector<std::thread*> threads(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    //     threads[i] = new std::thread(
    //         [=]() { PredictOne(&(*test_set)[i], &(*responses)[i]); });
    //     if (i % 100 == 0) {
    //       usleep(1000);  // 睡眠 1ms 避免过多线程创建
    //     }
    PredictOne(&(*test_set)[i], &(*responses)[i]);
  }
  //   for (int i = 0; i < num_samples; ++i) {
  //     if (threads[i]->joinable()) {
  //       threads[i]->join();
  //     }
  //     delete threads[i];
  //   }
}