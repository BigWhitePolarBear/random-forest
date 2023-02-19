#ifndef CARTREE_H_
#define CARTREE_H_

#include <CL/sycl.hpp>

#include "node.h"
#include "sample.h"

class Tree {
 public:
  /*
   *max_depth:                     最大深度
   *num_train_features_per_node:   每个节点的特征数
   *min_leaf_num_samples:          叶子样本少于多少时终止分类
   *min_info_gainn:                信息增益少于多少时终止分裂
   */
  Tree(int max_depth, int num_train_features_per_node, int min_leaf_num_samples,
       float min_info_gain, sycl::queue* q);
  ~Tree();
  void Train(SampleSet* samples);
  Result Predict(std::vector<float>* data);
  std::vector<Node*>* GetTreeArray() { return &cart_tree_array_; };
  void CreateNode(int id, int feature_idx, float threshold);
  void CreateLeaf(int id, float clas, float prob);

 private:
  sycl::queue* q_;
  int max_depth_;
  int num_nodes_;  // = 2^max_depth_-1
  int min_leaf_num_samples_;
  int num_train_features_per_node_;
  float min_info_gain_;
  std::vector<Node*>
      cart_tree_array_;  // 节点数组，通过下标*2+1或2的方式来记录父子节点关系
};
#endif  // CARTREE_H_
