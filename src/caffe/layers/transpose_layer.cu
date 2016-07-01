#include <cfloat>
#include <vector>

#include "caffe/layers/transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filter_map_layer_impl.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  FilterMapLayerImpl<Dtype>::Transpose(top[0]->mutable_gpu_data(),
				       bottom[0]->gpu_data(),
				       transpose_first_dim_,
				       bottom[0]->count() / transpose_first_dim_);
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //if (propagate_down[0]) {
  FilterMapLayerImpl<Dtype>::Transpose(bottom[0]->mutable_gpu_diff(),
				       top[0]->gpu_diff(),
				       top[0]->count() / transpose_first_dim_,
				       transpose_first_dim_);
    
  //}
}

INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);

}  // namespace caffe
