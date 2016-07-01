#include <cfloat>
#include <vector>

#include "caffe/layers/transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  transpose_axis_ = this->layer_param().transpose_param().transpose_axis();
}

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(transpose_axis_ > 0 && transpose_axis_ < bottom[0]->num_axes()) <<
    "Valid values of transpose_axis are [1..num_axis-1]";

  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape;

  transpose_first_dim_ = 1;
  for (int i = transpose_axis_; i < bottom_shape.size(); ++i) {
    top_shape.push_back(bottom_shape[i]);
  }
  for (int i = 0; i < transpose_axis_; ++i) {
    top_shape.push_back(bottom_shape[i]);
    transpose_first_dim_ *= bottom_shape[i];
  }

  top[0]->Reshape(top_shape);
}


template<typename Dtype>
void ref_trans(const Blob<Dtype>* in, Blob<Dtype>* out, int trans_dim1, bool diff) {
  int trans_dim2 = in->count() / trans_dim1;
  for (int i = 0; i < trans_dim1; ++i) {
    for (int j = 0; j < trans_dim2; ++j) {
      if (!diff) 
	out->mutable_cpu_data()[j*trans_dim1+i] = in->cpu_data()[i*trans_dim2+j];
      else
	out->mutable_cpu_diff()[j*trans_dim1+i] = in->cpu_diff()[i*trans_dim2+j];
    }
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ref_trans(bottom[0], top[0], transpose_first_dim_, false);
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  ref_trans(top[0], bottom[0], top[0]->count() / transpose_first_dim_, true);
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
