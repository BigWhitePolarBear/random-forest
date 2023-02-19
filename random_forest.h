#ifndef RANDOM_FOREST_H_
#define RANDOM_FOREST_H_

#include <vector>

#include "sample.h"
#include "tree.h"

class RandomForest {
 public:
  /*
   *num_trees:        树的数量
   *max_depth:        树的最大深度
   *min_leaf_sample:  叶子的样本数少于多少时，终止分裂
   *min_info_gain:    信息增益少于多少时，终止分裂
   */
  RandomForest(int num_trees, int max_depth, int min_leaf_sample,
               float min_info_gain);
  RandomForest(const char* model_path);
  ~RandomForest();
  /*
   *train_set:	  训练集
   *labels:       标签
   *num_samples:  样本数
   *num_features: 特征数
   *num_classes:  类别数
   */
  void Train(std::vector<std::vector<float>>* train_set,
             std::vector<float>* labels, int num_samples, int num_features,
             int num_classes, std::string mode);
  /*
   *sample:   一个样本
   *response: 预测结果
   */
  void PredictOne(std::vector<float>* sample, float* response);
  /*
   *test_set:   测试集
   *num_samples:样本数
   *responses:  预测结果
   */
  void Predict(std::vector<std::vector<float>>* test_set, int num_samples,
               std::vector<float>* responses);

 private:
  int num_train_samples_;            // 用来训练的样本数
  int num_test_samples_;             // 用来测试的样本数
  int num_features_;                 // 特征数
  int num_train_features_per_node_;  // 每个节点的特征数
  int num_trees_;                    // 树的数量
  int max_depth_;                    // 树的最大深度
  int num_classes_;                  // 类别数
  int min_leaf_num_samples_;  // 叶子的样本数少于多少时，终止分裂
  float min_info_gain_;  // 信息增益少于多少时，终止分裂
  std::vector<Tree*> trees_;  // 储存树引用的数组
  SampleSet*
      train_samples_;  // 数据集
};

#endif  // RANDOM_FOREST_H_
