#ifndef CAFFE_FILTER_MAP_LAYER_HPP_
#define CAFFE_FILTER_MAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class FilterMapLayer : public Layer<Dtype> {
 public:
  explicit FilterMapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
    main_conv_layer_.reset(new ConvolutionLayer<Dtype>(param));
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FilterMap"; }

  // Temp public for debugging
  void PostBackward();
  vector<shared_ptr<Blob<Dtype> > >& main_conv_blobs() {
    return main_conv_layer_->blobs();
  }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  int fmap_h_;
  int fmap_w_;
 private:
  // Internal convolution layer for sampling filterMap.
  shared_ptr<Layer<Dtype> > conv_layer_;

  // Holder for the top and bottom of the internal layer.
  vector<Blob<Dtype>*> conv_bottom_vec_;
  vector<Blob<Dtype>*> conv_top_vec_;

  // Internal conv layer for the regular convolution
  shared_ptr<ConvolutionLayer<Dtype> > main_conv_layer_;
  vector<Blob<Dtype>*> main_conv_bottom_vec_;
  vector<Blob<Dtype>*> main_conv_top_vec_;
  
  // Storage for the padded filterMap

  shared_ptr<Blob<Dtype> > gd_padding_buf_;
  shared_ptr<Blob<Dtype> > gd_final_buf_;

  // Storage for impl2
  shared_ptr<Blob<Dtype> > padding_buf_;
  shared_ptr<Blob<Dtype> > transpose_buf_;
  
  vector<shared_ptr<Blob<Dtype> > > hidden_blobs_;
};

}

#endif 
