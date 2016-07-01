#ifndef CAFFE_INDICATOR_LAYER_HPP_
#define CAFFE_INDICATOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {


/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class IndicatorDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit IndicatorDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~IndicatorDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IndicatorData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			   const vector<Blob<Dtype>*>& top);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleEntries();
  vector<int> idx1;
  vector<int> idx2;
  vector<float> score;

  std::vector<unsigned int> data_permutation_;
  int current_row;
};

} //namespace caffe

#endif
