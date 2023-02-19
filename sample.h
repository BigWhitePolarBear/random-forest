#ifndef SAMPLE_H_
#define SAMPLE_H_

#include <vector>

class SampleSet {
 public:
  static const int kSampleSelection = 1;
  static const int kFeatureSelection = 2;

  // 创建空数据集
  SampleSet(std::vector<std::vector<float>>* dataset,
            std::vector<float>* labels, int class_num, int num_samples,
            int num_features);
  // 复制
  SampleSet(SampleSet* samples);
  // 复制[start,end]的部分
  SampleSet(SampleSet* samples, int start, int end);
  ~SampleSet();
  // 有放回地抽样样本
  void RandomSelectSample(int num_samples, int num_selected_samples);
  // 无放回地抽样特征
  void RandomSelectFeature(int num_features, int num_selected_features);

  int GetNumClasses() { return num_classes_; };

  int GetNumSamples() { return num_samples_; };
  int GetNumFeatures() { return num_features_; };

  int GetNumSelectedSamples() { return num_selected_samples_; };
  int GetNumSelectedFeatures() { return num_selected_features_; };

  std::vector<int>* GetSampleIdxs() { return &sample_idxs_; };
  std::vector<int>* GetFeatureIdxs() { return &feature_idxs_; };

  std::vector<std::vector<float>>* dataset_;  // 数据集的内存引用
  std::vector<float>* labels_;                // 标签的内存引用

 private:
  std::vector<int> sample_idxs_;   // 采样出的样本下标
  std::vector<int> feature_idxs_;  // 采样出的特征下标
  int num_classes_;                // 类别数
  int num_features_;               // 特征数
  int num_samples_;                // 样本数
  int num_selected_samples_;       // 选中的样本数
  int num_selected_features_;      // 选中的特征数
};
#endif  // SAMPLE_H_
