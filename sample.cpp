#include "stdlib.h"

#include "sample.h"

SampleSet::SampleSet(std::vector<std::vector<float>>* dataset,
                     std::vector<float>* labels, int num_classes,
                     int num_samples, int num_features) {
  dataset_ = dataset;
  labels_ = labels;
  num_samples_ = num_samples;
  num_features_ = num_features;
  num_classes_ = num_classes;
  num_selected_samples_ = num_samples;
  num_selected_features_ = num_features;
}

SampleSet::SampleSet(SampleSet* samples) {
  dataset_ = samples->dataset_;
  labels_ = samples->labels_;
  num_classes_ = samples->GetNumClasses();
  num_features_ = samples->GetNumFeatures();
  num_samples_ = samples->GetNumSamples();
  num_selected_samples_ = samples->GetNumSelectedSamples();
  num_selected_features_ = samples->GetNumSelectedFeatures();
  sample_idxs_.resize(samples->GetSampleIdxs()->size());
  sample_idxs_ = *samples->GetSampleIdxs();
  feature_idxs_ = *samples->GetFeatureIdxs();
}

SampleSet::SampleSet(SampleSet* samples, int start, int end) {
  dataset_ = samples->dataset_;
  labels_ = samples->labels_;
  num_classes_ = samples->GetNumClasses();
  num_samples_ = samples->GetNumSamples();
  num_selected_samples_ = end - start;
  num_features_ = samples->GetNumFeatures();
  num_selected_features_ = samples->GetNumSelectedFeatures();
  sample_idxs_.resize(num_selected_samples_);
  sample_idxs_.assign(samples->GetSampleIdxs()->begin() + start,
                      samples->GetSampleIdxs()->begin() + end);
}

SampleSet::~SampleSet() {}

void SampleSet::RandomSelectSample(int num_samples, int num_selected_samples) {
  num_samples_ = num_samples;
  num_selected_samples_ = num_selected_samples;
  sample_idxs_.resize(num_selected_samples_);
  // 放回地抽样样本
  for (int i = 0; i < num_selected_samples; ++i) {
    sample_idxs_[i] = rand() % num_samples;
  }
}

void SampleSet::RandomSelectFeature(int num_features,
                                    int num_selected_features) {
  num_features_ = num_features;
  num_selected_features_ = num_selected_features;
  feature_idxs_.resize(num_features - num_selected_features);
  int i = 0, j = 0, k = 0, idx = 0;
  // 不放回地抽样特征
  for (i = 0, j = num_features - num_selected_features; j < num_features;
       ++j, ++i) {
    if (j == 0)
      idx = 0;
    else
      idx = rand() % j;
    bool flag = false;
    for (k = 0; k < i; ++k) {
      if (feature_idxs_[k] == idx) {
        flag = true;
        break;
      }
    }
    if (flag) {
      feature_idxs_[i] = j;
    } else {
      feature_idxs_[i] = idx;
    }
  }
}
