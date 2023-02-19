#ifndef NODE_H_
#define NODE_H_

#include <CL/sycl.hpp>

#include "sample.h"

struct Result {
  float label;  // label or value
  float prob;   // prob or 1
};

struct Pair {
  float feature;
  int idx;
};

class Node {
 public:
  Node(sycl::queue* q);
  ~Node();
  // 按特征下标对应的下标对样本下标进行排序
  void SortIndex(int feature_idx);
  SampleSet* samples_;  // 这个节点对应的数据集对象
  void SetLeaf(bool flag) { is_leaf_ = flag; };
  bool IsLeaf() { return is_leaf_; };
  void CalculateInfoGain(std::vector<Node*>* cartree_array, int id,
                         float min_info_gain);
  void CalculateParams();
  void CreateLeaf();
  int Predict(std::vector<float>* data, int id);
  void GetResult(Result& r);

  float GetClass() { return class_; };
  float GetProb() { return prob_; };

  void SetClass(float clas) { class_ = clas; };
  void SetProb(float prob) { prob_ = prob; };

  int GetFeatureIdxs() { return feature_idx_; };
  void SetFeatureIdx(int feature_index) { feature_idx_ = feature_index; };
  float GetThreshold() { return threshold_; };
  void SetThreshold(float threshold) { threshold_ = threshold; };

  float gini_;
  std::vector<float>* probs_;

 private:
  sycl::queue* q_;
  bool is_leaf_;
  int feature_idx_;
  float threshold_;
  float class_;  // 类别
  float prob_;   // 概率
};

#endif  // NODE_H_
