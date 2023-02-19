#include "node.h"

#include <algorithm>
#include <numeric>

Node::Node(sycl::queue* q) {
  q_ = q;
  is_leaf_ = false;
  feature_idx_ = -1;
  threshold_ = 0;
  samples_ = NULL;
  class_ = -1;
  prob_ = 0;
}

Node::~Node() {
  if (probs_ != NULL) {
    delete probs_;
    probs_ = NULL;
  }
}

void Node::SortIndex(int feature_idx) {
  std::vector<std::vector<float>>* data = samples_->dataset_;
  std::vector<int>* sample_idxs = samples_->GetSampleIdxs();
  int num_selected_samples = samples_->GetNumSelectedSamples();
  std::vector<Pair>* pairs = new std::vector<Pair>(num_selected_samples);
  q_->submit([&](sycl::handler &h) {
    h.parallel_for (sycl::range<1>(num_selected_samples), [=](sycl::id<1> i) {
      (*pairs)[i].idx = (*sample_idxs)[i];
      (*pairs)[i].feature = (*data)[(*sample_idxs)[i]][feature_idx];
    });
  }).wait_and_throw();
  std::sort(pairs->begin(), pairs->end(),
            [&](Pair a, Pair b) { return a.feature < b.feature; });
  q_->submit([&](sycl::handler &h) {
    h.parallel_for (sycl::range<1>(num_selected_samples), [=](sycl::id<1> i) {
      (*sample_idxs)[i] = (*pairs)[i].idx;
    });
  }).wait_and_throw();
  delete pairs;
//   std::vector<std::vector<float>>* data = samples_->dataset_;
//   std::vector<int>* sample_idxs = samples_->GetSampleIdxs();
//   std::vector<Pair> pairs(samples_->GetNumSelectedSamples());
//   for (int i = 0; i < samples_->GetNumSelectedSamples(); i++) {
//     pairs[i].idx = (*sample_idxs)[i];
//     pairs[i].feature = (*data)[(*sample_idxs)[i]][feature_idx];
//   }
//   std::sort(pairs.begin(), pairs.end(),
//             [&](Pair a, Pair b) { return a.feature < b.feature; });
//   for (int i = 0; i < samples_->GetNumSelectedSamples(); i++) {
//     (*sample_idxs)[i] = pairs[i].idx;
//   }  
}

void Node::CalculateParams() {
  std::vector<int>* sample_idxs = samples_->GetSampleIdxs();
  int num_samples = samples_->GetNumSelectedSamples();
  int num_classes = samples_->GetNumClasses();
  probs_ = new std::vector<float>(num_classes, 0);
  std::vector<float>* probs = probs_;
  SampleSet* samples = samples_;
  q_->submit([&](sycl::handler &h) {
    h.parallel_for (sycl::range<1>(num_samples), [=](sycl::id<1> i) {
      (*probs)[static_cast<int>((*(samples->labels_))[(*sample_idxs)[i]])]++;
    });
  }).wait_and_throw();
  std::vector<float>* ps = new std::vector<float>(num_classes);
  q_->submit([&](sycl::handler &h) {
    h.parallel_for (sycl::range<1>(num_classes), [=](sycl::id<1> i) {
      (*ps)[i] = (*probs)[i] / num_samples;
    });
  }).wait_and_throw();
  gini_ = 1.;
  for (int i = 0; i < num_classes; i++) {
    gini_ -= (*ps)[i] * (*ps)[i];
  }
  delete ps;
}

void Node::CalculateInfoGain(std::vector<Node*>* node_array, int idx,
                             float min_info_gain) {
  // 临时所需的资源
  int i = 0, j = 0, k = 0;
  std::vector<int>* sample_idxs = samples_->GetSampleIdxs();
  std::vector<int>* feature_idxs = samples_->GetFeatureIdxs();
  std::vector<std::vector<float>>* data = samples_->dataset_;
  std::vector<float>* labels = samples_->labels_;
  int num_features = samples_->GetNumSelectedFeatures();
  int num_samples = samples_->GetNumSelectedSamples();
  int num_classes = samples_->GetNumClasses();
  // 最后储存的信息
  float max_info_gain = 0;
  int max_feature_idx = 0;
  float max_threshold = 0;
  float max_gini_left = 0;
  float max_gini_right = 0;
  int max_samples_on_left = 0;
  std::vector<float>* max_probs_left = new std::vector<float>(num_classes, 0);
  std::vector<float>* max_probs_right = new std::vector<float>(num_classes, 0);
  // 记录循环中每一个特征对应的信息
  float f_max_info_gain = 0;
  int f_max_feature_idx = 0;
  float f_max_threshold = 0;
  float f_max_gini_left = 0;
  float f_max_gini_right = 0;
  int f_max_samples_on_left = 0;
  std::vector<float>* f_max_probs_left = new std::vector<float>(num_classes);
  std::vector<float>* f_max_probs_right = new std::vector<float>(num_classes);
  // 循环中临时变量
  float gini_left = 0, gini_right = 0, info_gain = 0;
  std::vector<float>* probs_left = new std::vector<float>(num_classes);
  std::vector<float>* probs_right = new std::vector<float>(num_classes);
  for (i = 0; i < num_features; ++i) {
    f_max_info_gain = 0;
    f_max_feature_idx = (*feature_idxs)[i];
    f_max_gini_left = 0;
    f_max_gini_right = 0;
    f_max_threshold = 0;
    f_max_samples_on_left = 0;
    f_max_probs_left->assign(num_classes, 0);
    f_max_probs_right->assign(num_classes, 0);
    // 按当前的特征进行排序
    SortIndex((*feature_idxs)[i]);
    // 初始化左右概率
    probs_left->assign(num_classes, 0);
    *probs_right = *probs_;
    for (j = 0; j < num_samples; ++j) {
      gini_left = 0;
      gini_right = 0;
      info_gain = 0;
      (*probs_left)[static_cast<int>((*labels)[(*sample_idxs)[j]])]++;
      (*probs_right)[static_cast<int>((*labels)[(*sample_idxs)[j]])]--;
      // 跳过相近的特征，避免过拟合
      if (j < num_samples - 1) {
        if (((*data)[(*sample_idxs)[j + 1]][(*feature_idxs)[i]] -
             (*data)[(*sample_idxs)[j]][(*feature_idxs)[i]]) < 0.000001) {
          continue;
        }
      }
//       for (k = 0; k < num_classes; k++) {
//         float p = (*probs_left)[k] / (j + 1);
//         gini_left += (p * p);
//       }
//       gini_left = 1 - gini_left;
//       for (k = 0; k < num_classes; k++) {
//         float p = (*probs_right)[k] / (num_samples - j - 1);
//         gini_right += (p * p);
//       }
//       gini_right = 1 - gini_right;
      std::vector<float>* ps = new std::vector<float>(num_classes);
      q_->submit([&](sycl::handler &h) {
        h.parallel_for (sycl::range<1>(num_classes), [=](sycl::id<1> k) {
          (*ps)[k] = (*probs_left)[k] / (j + 1);
        });
      }).wait_and_throw();
      gini_left = 1;
      for (int k = 0; k < num_classes; k++) {
        gini_left -= (*ps)[k] * (*ps)[k];
      }
      q_->submit([&](sycl::handler &h) {
        h.parallel_for (sycl::range<1>(num_classes), [=](sycl::id<1> k) {
          (*ps)[k] = (*probs_right)[k] / ((num_samples - j - 1));
        });
      }).wait_and_throw();
      gini_right = 1;
      for (int k = 0; k < num_classes; k++) {
        gini_right -= (*ps)[k] * (*ps)[k];
      }
      delete ps;
      float left_ratio = (j + 1.0) / num_samples;
      float right_ratio = (num_samples - j - 1.0) / num_samples;
      info_gain = gini_ - left_ratio * gini_left - right_ratio * gini_right;
      if (info_gain > f_max_info_gain) {
        f_max_info_gain = info_gain;
        f_max_gini_left = gini_left;
        f_max_gini_right = gini_right;
        f_max_threshold = ((*data)[(*sample_idxs)[j]][(*feature_idxs)[i]] +
                           (*data)[(*sample_idxs)[j + 1]][(*feature_idxs)[i]]) /
                          2;
        f_max_samples_on_left = j;
        *f_max_probs_left = *probs_left;
        *f_max_probs_right = *probs_right;
      }
    }
    if (f_max_info_gain > max_info_gain) {
      max_info_gain = f_max_info_gain;
      max_gini_left = f_max_gini_left;
      max_gini_right = f_max_gini_right;
      max_feature_idx = f_max_feature_idx;
      max_threshold = f_max_threshold;
      max_samples_on_left = f_max_samples_on_left;
      swap(max_probs_left, f_max_probs_left);
      swap(max_probs_right, f_max_probs_right);
    }
  }
  SortIndex(max_feature_idx);
  if (max_info_gain < min_info_gain) {
    CreateLeaf();
  } else {
    feature_idx_ = max_feature_idx;
    threshold_ = max_threshold;
    (*node_array)[idx * 2 + 1] = new Node(q_);
    (*node_array)[idx * 2 + 2] = new Node(q_);
    (*node_array)[idx * 2 + 1]->gini_ = max_gini_left;
    (*node_array)[idx * 2 + 1]->probs_ = max_probs_left;
    (*node_array)[idx * 2 + 2]->gini_ = max_gini_right;
    (*node_array)[idx * 2 + 2]->probs_ = max_probs_right;
    // 给左右子节点分配数据集的引用
    SampleSet* left_samples = new SampleSet(samples_, 0, max_samples_on_left);
    SampleSet* right_samples =
        new SampleSet(samples_, max_samples_on_left + 1, num_samples - 1);
    (*node_array)[idx * 2 + 1]->samples_ = left_samples;
    (*node_array)[idx * 2 + 2]->samples_ = right_samples;
  }
  // 释放内存
  delete probs_;
  probs_ = NULL;
  delete f_max_probs_left;
  delete f_max_probs_right;
  delete probs_left;
  delete probs_right;
}

void Node::CreateLeaf() {
  class_ = 0;
  prob_ = (*probs_)[0];
  for (int i = 1; i < samples_->GetNumClasses(); ++i) {
    if ((*probs_)[i] > prob_) {
      class_ = i;
      prob_ = (*probs_)[i];
    }
  }
  prob_ /= samples_->GetNumSelectedSamples();
  is_leaf_ = true;
}

int Node::Predict(std::vector<float>* data, int id) {
  if ((*data)[feature_idx_] < threshold_) {
    return id * 2 + 1;
  } else {
    return id * 2 + 2;
  }
}

void Node::GetResult(Result& r) {
  r.label = class_;
  r.prob = prob_;
}
