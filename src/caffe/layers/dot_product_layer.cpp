#include <cfloat>
#include <vector>

#include "caffe/layers/dot_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DotProductLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  // Check the num of the two bottoms equal.
  int _axes_a, _axes_b;
  vector<int> dot_shape;
  switch (this->layer_param_.dot_product_param().mode()) {
    case DotProductParameter_Mode_VV:
      CHECK_EQ(bottom[0]->shape(-1), bottom[1]->shape(-1));
      dot_shape.push_back(1);
      _axes_a = 1;
      _axes_b = 1;
      break;
    case DotProductParameter_Mode_VM:
      CHECK_EQ(bottom[0]->shape(-1), bottom[1]->shape(-2));
      dot_shape.push_back(bottom[1]->shape(-1));
      _axes_a = 1;
      _axes_b = 2;
      break;
    case DotProductParameter_Mode_MV:
      CHECK_EQ(bottom[0]->shape(-1), bottom[1]->shape(-1));
      dot_shape.push_back(bottom[0]->shape(-2));
      _axes_a = 2;
      _axes_b = 1;
      break;
    case DotProductParameter_Mode_MM:
      CHECK_EQ(bottom[0]->shape(-1), bottom[1]->shape(-2));
      dot_shape.push_back(bottom[0]->shape(-2));
      dot_shape.push_back(bottom[1]->shape(-1));
      _axes_a = 2;
      _axes_b = 2;
      break;
    default:
      LOG(FATAL) << "Unknown mode enum value ";
  }

  CHECK_EQ(bottom[0]->num_axes() - _axes_a, bottom[1]->num_axes() - _axes_b)
      << "The number of axes not agree: " << bottom[0]->num_axes() - _axes_a
      << " vs. " << bottom[1]->num_axes() - _axes_b;

  vector<int> top_shape;
  for (int i = 0; i < bottom[0]->num_axes() - _axes_a; ++i) {
    CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
    top_shape.push_back(bottom[0]->shape(i));
  }

  top_shape.insert(top_shape.end(), dot_shape.begin(), dot_shape.end());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Forward_cpu_GEMV_L(const vector<Blob<Dtype> *> &bottom,
                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data_a = bottom[0]->cpu_data();
  const Dtype *bottom_data_b = bottom[1]->cpu_data();
  const int num = bottom[0]->count(0, bottom[0]->num_axes() - 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int _m = bottom[1]->shape(-2);
  const int _n = bottom[1]->shape(-1);
  const int _stride_a = _m;
  const int _stride_b = _m * _n;
  const int _stride_c = _n;
  for (int i = 0; i < num; ++i) {
    caffe_cpu_gemv(CblasTrans, _m, _n, (Dtype)1, bottom_data_b + i * _stride_b,
                   bottom_data_a + i * _stride_a, (Dtype)0.,
                   top_data + i * _stride_c);
  }
}
template <typename Dtype>
void DotProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  switch (this->layer_param_.dot_product_param().mode()) {
    case DotProductParameter_Mode_VM:
      Forward_cpu_GEMV_L(bottom, top);
      break;
    default:
      NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void Backward_cpu_GEMV_L(const vector<Blob<Dtype> *> &top,
                         const vector<bool> &propagate_down,
                         const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *bottom_data_a = bottom[0]->cpu_data();
  const Dtype *bottom_data_b = bottom[1]->cpu_data();
  const int num = bottom[0]->count(0, bottom[0]->num_axes() - 1);

  const int _m = bottom[1]->shape(-2);
  const int _n = bottom[1]->shape(-1);
  const int _stride_a = _m;
  const int _stride_b = _m * _n;
  const int _stride_c = _n;

  if (propagate_down[0]) {
    Dtype *bottom_diff_a = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_cpu_gemv(CblasNoTrans, _m, _n, (Dtype)1.,
                     bottom_data_b + i * _stride_b, top_diff + i * _stride_c,
                     (Dtype)0., bottom_diff_a + i * _stride_a);
    }
  }
  if (propagate_down[1]) {
    Dtype *bottom_diff_b = bottom[1]->mutable_cpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            _m,                             // M
                            _n,                             // N
                            1,                              // K
                            (Dtype)1,                       // alpha
                            bottom_data_a + i * _stride_a,  // x
                            top_diff + i * _stride_c,       // y
                            (Dtype)0,                       // beta
                            bottom_diff_b + i * _stride_b   // a
                            );
    }
  }
}

template <typename Dtype>
void DotProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
  switch (this->layer_param_.dot_product_param().mode()) {
    case DotProductParameter_Mode_VM:
      Backward_cpu_GEMV_L(top, propagate_down, bottom);
      break;
    default:
      NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotProductLayer);
#endif

INSTANTIATE_CLASS(DotProductLayer);
REGISTER_LAYER_CLASS(DotProduct);
}  // namespace caffe
