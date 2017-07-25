// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layers/filter_map_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filter_map_layer_impl.hpp"
#include <cmath>

namespace caffe {

  
template <typename Dtype>
void LocalFilterMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LocalFilterMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LocalFilterMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

  
template <typename Dtype>
void LocalFilterMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
  // First setup the normal convolution stuff.

  this->main_conv_layer_->SetUp(bottom, top);
  // TODO: handle the case with B.
  // I think a shared global bias makes more sense.

  // Shape of the filtermap: channels * (fw * fh),
  // where fw == fh and fw*fh = num_output_ is assumed
  FilterMapParameter fmap_param = this->layer_param_.filter_map_param();

  this->fmap_h_ = fmap_param.map_height();
  this->fmap_w_ = fmap_param.map_width();

  CHECK_EQ(round(fmap_h_*fmap_w_*4/9), main_conv_layer_->num_output_);
  
  CHECK_EQ(main_conv_layer_->kernel_shape_.count(), 2) <<  "Only support 2d kernel.";
  CHECK_EQ(main_conv_layer_->kernel_shape_.cpu_data()[0], main_conv_layer_->kernel_shape_.cpu_data()[1]) <<  "Only support square kernel.";

  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];
  
  vector<int> filtermap_shape(2);

  DLOG(ERROR) << "shape of filtermap: " << main_conv_layer_->channels_ << "," << main_conv_layer_->num_output_;
  filtermap_shape[0] = main_conv_layer_->channels_;
  filtermap_shape[1] = main_conv_layer_->num_output_;
 
  this->blobs_.resize(2);
  // FilterMap
  this->blobs_[0].reset(new Blob<Dtype>(filtermap_shape));

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
							    this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  
  // BiasTerm
  this->blobs_[1] = main_conv_layer_->blobs()[1];


  vector<int> gd_final_shape(4);
  gd_final_shape[0] = main_conv_layer_->channels_;
  gd_final_shape[1] = 1;
  gd_final_shape[2] = fmap_h_;
  gd_final_shape[3] = fmap_w_;

  fm_gd_blob.reset(new Blob<Dtype>(gd_final_shape));
  
  main_conv_bottom_vec_.clear();
  main_conv_bottom_vec_.push_back(bottom[0]);
  main_conv_top_vec_.clear();
  main_conv_top_vec_.push_back(top[0]);
  
  
  
  CHECK_EQ(gd_final_buf_->shape(0), gd_final_shape[0]);
  CHECK_EQ(gd_final_buf_->shape(1), gd_final_shape[1]);
  CHECK_EQ(gd_final_buf_->shape(2), gd_final_shape[2]);
  CHECK_EQ(gd_final_buf_->shape(3), gd_final_shape[3]);
}

template <typename Dtype>
void LocalFilterMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];
  // Padding and sampling the filtermap

  FilterMapLayerImpl<Dtype>::ExtractFilters_3x3(
                   		    main_conv_layer_->blobs()[0]->mutable_gpu_data(),
		            this->blobs_[0]->gpu_data(),
				    fmap_h_,
				    fmap_w_,
				    main_conv_layer_->channels_,
				    patchSize);

  //Start debug"
  //DLOG(ERROR) << "FilterMaps asum: " << this->blobs_[0]->asum_data();
  //DLOG(ERROR) << "PaddedBuf asum: " << padding_buf_->asum_data();
  //DLOG(ERROR) << "Filters asum: " << main_conv_layer_->blobs()[0]->asum_data();
  
  
  //End debug"
  // Using the sampled filters to do regular convolution.
  main_conv_layer_->Forward(main_conv_bottom_vec_, main_conv_top_vec_);
}

template <typename Dtype>
void LocalFilterMapLayer<Dtype>::PostBackward() {
  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];

  FilterMapLayerImpl<Dtype>::AggregateGradForFilterMap(
				       this->fm_gd_blob,
				       main_conv_layer_->blobs()[0]->gpu_diff(),
				       fmap_h_,
				       fmap_w_,
				       main_conv_layer_->channels_,
				       patchSize);


  this->conv_layer_->Forward(conv_bottom_vec_,
			     conv_top_vec_);

  // Copyt from the 'data' of fm_gd_blob to the 'diff' of blobs_[2]
  if (fm_gd_blob->count() != this->blobs_[0]->count()) {
    LOG(FATAL) << "GD_FINAL_BUF DIM NOT AGREE WITH FILTERMAP";
  }
  int count = fm_gd_blob->count();
  caffe_copy(count, fm_gd_blob->gpu_data(),
	     static_cast<Dtype*>(this->blobs_[0]->mutable_gpu_diff()));
  // Done!
}

template <typename Dtype>
void LocalFilterMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool> &propagate_down, const vector<Blob<Dtype>*>& bottom) {


  // Step 1: Normal convolution layer gradient.
  caffe_gpu_set(main_conv_layer_->blobs_[0]->count(), Dtype(0), main_conv_layer_->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_set(main_conv_layer_->blobs_[1]->count(), Dtype(0), main_conv_layer_->blobs_[1]->mutable_gpu_diff());
  main_conv_layer_->Backward(main_conv_top_vec_, propagate_down, main_conv_bottom_vec_);

  // Step 2: backward the gradients to padded filtermap
  PostBackward();
}

#endif //IMP2

INSTANTIATE_CLASS(LocalFilterMapLayer);
REGISTER_LAYER_CLASS(LocalFilterMap);
}  // namespace caffe
