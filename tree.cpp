#include "tree.h"

Tree::Tree(int max_depth, int num_train_features_per_node,
           int min_leaf_num_samples, float min_info_gain, sycl::queue* q) {
  max_depth_ = max_depth;
  num_train_features_per_node_ = num_train_features_per_node;
  min_leaf_num_samples_ = min_leaf_num_samples;
  min_info_gain_ = min_info_gain;
  num_nodes_ = static_cast<int>(pow(2.0, max_depth_) - 1);
  cart_tree_array_.resize(num_nodes_);
  cart_tree_array_.assign(num_nodes_, NULL);
  q_ = q;
}

Tree::~Tree() {
  for (int i = 0; i < num_nodes_; ++i) {
    if (cart_tree_array_[i] != NULL) {
      delete cart_tree_array_[i];
      cart_tree_array_[i] = NULL;
    }
  }
}

Result Tree::Predict(std::vector<float>* data) {
  int position = 0;
  Node* head = cart_tree_array_[position];
  while (!head->IsLeaf()) {
    position = head->Predict(data, position);
    head = cart_tree_array_[position];
  }
  Result r;
  head->GetResult(r);
  return r;
}

void Tree::Train(SampleSet* samples) {
  // 初始化根节点，并随机采样样本
  std::vector<int> feature_idxs(num_train_features_per_node_);
  SampleSet* node_samples =
      new SampleSet(samples, 0, samples->GetNumSelectedSamples());
  cart_tree_array_[0] = new Node(q_);
  cart_tree_array_[0]->samples_ = node_samples;
  // 计算参数
  cart_tree_array_[0]->CalculateParams();
  for (int i = 0; i < num_nodes_; ++i) {
    int parent_idx = (i - 1) / 2;
    // 没有父节点，跳过
    if (cart_tree_array_[parent_idx] == NULL) {
      continue;
    }
    // 为叶子节点，跳过
    if (i > 0 && cart_tree_array_[parent_idx]->IsLeaf()) {
      continue;
    }
    // 到达最大深度（左子树要超限），创建叶子并继续
    if (i * 2 + 1 >= num_nodes_) {
      cart_tree_array_[i]->CreateLeaf();
      continue;
    }
    // 当前样本数未小于阈值，创建叶子并继续
    if (cart_tree_array_[i]->samples_->GetNumSelectedSamples() <=
        min_leaf_num_samples_) {
      cart_tree_array_[i]->CreateLeaf();
      continue;
    }
    cart_tree_array_[i]->samples_->RandomSelectFeature(
        samples->GetNumFeatures(), num_train_features_per_node_);
    // 计算信息增益，决定是否要继续分裂
    cart_tree_array_[i]->CalculateInfoGain(&cart_tree_array_, i,
                                           min_info_gain_);
  }
  delete node_samples;
  node_samples = NULL;
}

void Tree::CreateNode(int idx, int feature_idx, float threshold) {
  cart_tree_array_[idx] = new Node(q_);
  cart_tree_array_[idx]->SetLeaf(false);
  cart_tree_array_[idx]->SetFeatureIdx(feature_idx);
  cart_tree_array_[idx]->SetThreshold(threshold);
}

void Tree::CreateLeaf(int idx, float clas, float prob) {
  cart_tree_array_[idx] = new Node(q_);
  cart_tree_array_[idx]->SetLeaf(true);
  cart_tree_array_[idx]->SetClass(clas);
  cart_tree_array_[idx]->SetProb(prob);
}