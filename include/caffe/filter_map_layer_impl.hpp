#ifndef FILTER_MAP_LAYER_IMPL_HPP_
#define FILTER_MAP_LAYER_IMPL_HPP_

#include "caffe/blob.hpp"

namespace caffe{

template <typename Dtype>
class FilterMapLayerImpl {

  // FilterMapLayerImpl(_filterMapLayer):filterMapLayer(_filterMapLayer){}

  //extract filters from filterMapPad_weight (device pointer) to weight (device pointer)
  //f_h,f_w: height and width of the filter map
  //ch = channels_ / group_;
  //kernel_size_ kernel size
public:  
  static void ExtractFilters(Dtype* weight, Dtype* filterMapPad_weight, const Dtype* filterMap_weight, int f_h, int f_w, int ch, int kernel_size_);


  //obtain weight_diff_tran_pad_blob and conv_sqfilter, the convolutoin of the two is the diff of the filtermap
  //weight_diff the diff of the sampled filters from the filter map dimension is (f_h*f_w)*ch*kernel_size_*kernel_size_
  //f_h,f_w: height and width of the filter map
  //ch = channels_ / group_;
  //kernel_size_ kernel size 

  static void TransPadFilterGd(shared_ptr<Blob<Dtype> > weight_diff_tran_pad_blob, shared_ptr<Blob<Dtype> > conv_sqfilter, const Dtype* weight_diff, int f_h, int f_w, int ch, int kernel_size_);


  static void Padding(Dtype* outdata, const Dtype *indata, int f_h, int f_w, int ch, int kernel_size);
  static void UnPadding(Dtype* outdata, Dtype *indata, int f_h, int f_w, int ch, int kernel_size);
  static void Transpose(Dtype *outdata, const Dtype *indata, int h, int w); 
};
  
}

#endif //FILTER_MAP_LAYER_IMPLHPP_
