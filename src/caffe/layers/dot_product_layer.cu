
#include <vector>

#include "caffe/layers/dot_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Forward_gpu_GEMV_L(const vector<Blob<Dtype> *> &bottom,
                        const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data_a = bottom[0]->gpu_data();
  const Dtype *bottom_data_b = bottom[1]->gpu_data();
  const int num = bottom[0]->count(0, bottom[0]->num_axes() - 1);
  Dtype *top_data = top[0]->mutable_gpu_data();
  const int _m = bottom[1]->shape(-2);
  const int _n = bottom[1]->shape(-1);
  const int _stride_a = _m;
  const int _stride_b = _m * _n;
  const int _stride_c = _n;
  for (int i = 0; i < num; ++i) {
    caffe_gpu_gemv(CblasTrans, _m, _n, (Dtype)1, bottom_data_b + i * _stride_b,
                   bottom_data_a + i * _stride_a, (Dtype)0.,
                   top_data + i * _stride_c);
  }
}

template <typename Dtype>
void DotProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  switch (this->layer_param_.dot_product_param().mode()) {
    case DotProductParameter_Mode_VM:
      Forward_gpu_GEMV_L(bottom, top);
      break;
    default:
      NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void Backward_gpu_GEMV_L(const vector<Blob<Dtype> *> &top,
                         const vector<bool> &propagate_down,
                         const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->gpu_diff();
  const Dtype *bottom_data_a = bottom[0]->gpu_data();
  const Dtype *bottom_data_b = bottom[1]->gpu_data();
  const int num = bottom[0]->count(0, bottom[0]->num_axes() - 1);

  const int _m = bottom[1]->shape(-2);
  const int _n = bottom[1]->shape(-1);
  const int _stride_a = _m;
  const int _stride_b = _m * _n;
  const int _stride_c = _n;

  if (propagate_down[0]) {
    Dtype *bottom_diff_a = bottom[0]->mutable_gpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_gpu_gemv(CblasNoTrans, _m, _n, (Dtype)1.,
                     bottom_data_b + i * _stride_b, top_diff + i * _stride_c,
                     (Dtype)0., bottom_diff_a + i * _stride_a);
    }
  }
  if (propagate_down[1]) {
    Dtype *bottom_diff_b = bottom[1]->mutable_gpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
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
void DotProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
  switch (this->layer_param_.dot_product_param().mode()) {
    case DotProductParameter_Mode_VM:
      Backward_gpu_GEMV_L(top, propagate_down, bottom);
      break;
    default:
      NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductLayer);

}  // namespace caffe
