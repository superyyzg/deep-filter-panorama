// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layers/filter_map_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filter_map_layer_impl.hpp"


namespace caffe {

  //#define IMPL2
#define BLOB_STORE hidden_blobs_;
  
#ifdef IMPL2
template <typename Dtype>
void generate_sampling_filter(const int ch,
			      const int map_size,
			      const int patch_size,
			      caffe::Blob<Dtype> *filter) {
  
  std::vector<int> filterShape(4);
  filterShape[0] = patch_size * patch_size;
  filterShape[1] = ch;
  filterShape[2] = patch_size;
  filterShape[3] = patch_size;

  filter->Reshape(filterShape);

  Dtype *filter_data = filter->mutable_cpu_data();
  caffe_set<Dtype>(filter->count(), 0, filter_data);

  for (int i = 0; i < filterShape[0]; ++i) {
    for (int ich = 0; ich < ch; ++ich) {
      int offset = filter->offset(i, ich);
      offset = offset + i;
      filter_data[offset] = 1;
    }
  }
}


template <typename Dtype>
void FilterMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FilterMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
  // First setup the normal convolution stuff.
  this->main_conv_layer_->SetUp(bottom, top);  
  // TODO: handle the case with B.
  // I think a shared global bias makes more sense.


  // Shape of the filtermap: channels * (fw * fh),
  // where fw == fh and fw*fh = num_output_ is assumed

  int fmap_size = int(sqrt(main_conv_layer_->num_output_));
  CHECK_EQ(fmap_size*fmap_size, main_conv_layer_->num_output_);
  CHECK_EQ(main_conv_layer_->kernel_shape_.count(), 2) <<  "Only support 2d kernel.";
  CHECK_EQ(main_conv_layer_->kernel_shape_.cpu_data()[0], main_conv_layer_->kernel_shape_.cpu_data()[1]) <<  "Only support square kernel.";

  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];


  vector<int> filtermap_shape(2);

  DLOG(ERROR) << "shape of filtermap: " << main_conv_layer_->channels_ << "," << main_conv_layer_->num_output_;
  filtermap_shape[0] = main_conv_layer_->channels_;
  filtermap_shape[1] = main_conv_layer_->num_output_;
 
  this->blobs_.resize(1);
  // FilterMap
  this->blobs_[0].reset(new Blob<Dtype>(filtermap_shape));
  // BiasTerm
  //this->blobs_[1] = main_conv_layer_->blobs()[1];

  // Initialize the buffer for padded map.
  vector<int> padding_shape(4);
  padding_shape[0] = filtermap_shape[0];
  padding_shape[1] = 1;
  padding_shape[2] = fmap_size + patchSize - 1;
  padding_shape[3] = fmap_size + patchSize - 1;
  padding_buf_.reset(new Blob<Dtype>(padding_shape));

  vector<int> transpose_shape(4);
  transpose_shape[0] = main_conv_layer_->channels_;
  transpose_shape[1] = patchSize*patchSize;
  transpose_shape[2] = fmap_size;
  transpose_shape[3] = fmap_size;
  transpose_buf_.reset(new Blob<Dtype>(transpose_shape));


  // TODO The internal conv layer.

  conv_bottom_vec_.clear();
  conv_bottom_vec_.push_back(padding_buf_.get());

  conv_top_vec_.clear();
  conv_top_vec_.push_back(transpose_buf_.get());

  LayerParameter maps_sampling_layer_param;
  maps_sampling_layer_param.set_type("Convolution");
  ConvolutionParameter *maps_sampling_layer_conv_param = maps_sampling_layer_param.mutable_convolution_param();
  maps_sampling_layer_conv_param->add_kernel_size(patchSize);
  maps_sampling_layer_conv_param->set_num_output(patchSize * patchSize);

  this->conv_layer_ = LayerRegistry<Dtype>::CreateLayer(maps_sampling_layer_param);
  this->conv_layer_->SetUp(conv_bottom_vec_, conv_top_vec_);

  generate_sampling_filter(main_conv_layer_->channels_,
			    fmap_size,
			    patchSize,
			   conv_layer_->blobs()[0].get());
  
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int fmap_size = int(sqrt(main_conv_layer_->num_output_));
  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];
  
  // Padding the filtermap
  FilterMapLayerImpl<Dtype>::Padding(
		  padding_buf_->mutable_gpu_data(),
		  this->blobs_[0]->gpu_data(),
		   fmap_size,
	           fmap_size,	
		  main_conv_layer_->channels_,
		  patchSize);
  
  // Sample the actual filters from filtermap
  conv_layer_->Forward(conv_bottom_vec_,
				   conv_top_vec_);

  // Transpose the sampled filters matrix

  FilterMapLayerImpl<Dtype>::Transpose(main_conv_layer_->blobs_[0]->mutable_gpu_data(),
                       transpose_buf_->gpu_data(),
	         	transpose_buf_->count() / fmap_size / fmap_size,
             		fmap_size * fmap_size);

  //DLOG(ERROR) << "FilterMaps asum: " << this->blobs_[0]->asum_data();
  //DLOG(ERROR) << "PaddedBuf asum: " << padding_buf_->asum_data();
  //DLOG(ERROR) << "Filters asum: " << main_conv_layer_->blobs()[0]->asum_data();

  // Using the sampled filters to do regular convolution.
  main_conv_layer_->Forward(bottom,top);
}

template <typename Dtype>
void FilterMapLayer<Dtype>::PostBackward() {
   int fmap_size = int(sqrt(main_conv_layer_->num_output_));
   const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];

  // Step 3: Transpose the gradient back
  FilterMapLayerImpl<Dtype>::Transpose(transpose_buf_->mutable_gpu_diff(),
		main_conv_layer_->blobs_[0]->gpu_diff(), 
		fmap_size * fmap_size,
		transpose_buf_->count() / fmap_size / fmap_size);
  
  // Step 3: "Unpad" the filterMap
  // Step 2: backward the gradients to padded filtermap
  vector<bool> p_down(1);
  p_down[0] = true;
  conv_layer_->Backward(conv_top_vec_,
				     p_down,
				     conv_bottom_vec_);

  FilterMapLayerImpl<Dtype>::UnPadding(
                    this->blobs_[0]->mutable_gpu_diff(),
		    padding_buf_->mutable_gpu_diff(),
		    fmap_size,
		    fmap_size,
		    main_conv_layer_->channels_,
		    patchSize);
  // Done:-)

}

template <typename Dtype>
void FilterMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool> &propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
  // Step 1: Normal convolution layer gradient.
  main_conv_layer_->Backward(top, propagate_down, bottom);

}


#else
  
template <typename Dtype>
void FilterMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

  
template <typename Dtype>
void FilterMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  CHECK_EQ(fmap_h_*fmap_w_, main_conv_layer_->num_output_);
  
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

  // Initialize the buffer for padded map.
  vector<int> padding_shape(4);
  padding_shape[0] = 1;
  padding_shape[1] = filtermap_shape[0];
  padding_shape[2] = fmap_h_ + patchSize - 1;
  padding_shape[3] = fmap_w_ + patchSize - 1;
  padding_buf_.reset(new Blob<Dtype>(padding_shape));

  vector<int> gd_padding_shape(4);
  gd_padding_shape[0] = main_conv_layer_->channels_;
  gd_padding_shape[1] = patchSize*patchSize;
  gd_padding_shape[2] = fmap_h_ + patchSize - 1;
  gd_padding_shape[3] = fmap_w_ + patchSize - 1;
  gd_padding_buf_.reset(new Blob<Dtype>(gd_padding_shape));

  vector<int> gd_final_shape(4);
  gd_final_shape[0] = main_conv_layer_->channels_;
  gd_final_shape[1] = 1;
  gd_final_shape[2] = fmap_h_;
  gd_final_shape[3] = fmap_w_;

  gd_final_buf_.reset(new Blob<Dtype>(gd_final_shape));
  
  
  conv_bottom_vec_.clear();
  conv_bottom_vec_.push_back(gd_padding_buf_.get());

  conv_top_vec_.clear();
  conv_top_vec_.push_back(gd_final_buf_.get());

  main_conv_bottom_vec_.clear();
  main_conv_bottom_vec_.push_back(bottom[0]);
  main_conv_top_vec_.clear();
  main_conv_top_vec_.push_back(top[0]);
  
  LayerParameter conv_layer_param;
  conv_layer_param.set_type("Convolution");
  ConvolutionParameter *conv_param = conv_layer_param.mutable_convolution_param();
  conv_param->add_kernel_size(patchSize);
  conv_param->set_num_output(1);
  conv_param->set_bias_term(false);
  this->conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_layer_param);

  this->conv_layer_->SetUp(conv_bottom_vec_, conv_top_vec_);
  
  CHECK_EQ(gd_final_buf_->shape(0), gd_final_shape[0]);
  CHECK_EQ(gd_final_buf_->shape(1), gd_final_shape[1]);
  CHECK_EQ(gd_final_buf_->shape(2), gd_final_shape[2]);
  CHECK_EQ(gd_final_buf_->shape(3), gd_final_shape[3]);
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];
  // Padding and sampling the filtermap

  FilterMapLayerImpl<Dtype>::ExtractFilters(
                   		    main_conv_layer_->blobs()[0]->mutable_gpu_data(),
				    padding_buf_->mutable_gpu_data(),
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
void FilterMapLayer<Dtype>::PostBackward() {
  const int patchSize = main_conv_layer_->kernel_shape_.cpu_data()[0];

  FilterMapLayerImpl<Dtype>::TransPadFilterGd(
				       this->gd_padding_buf_,
				       this->conv_layer_->blobs()[0],
				       main_conv_layer_->blobs()[0]->gpu_diff(),
				       fmap_h_,
				       fmap_w_,
				       main_conv_layer_->channels_,
				       patchSize);


  this->conv_layer_->Forward(conv_bottom_vec_,
			     conv_top_vec_);

  // Copyt from the 'data' of gd_final_buf_ to the 'diff' of blobs_[2]
  if (gd_final_buf_->count() != this->blobs_[0]->count()) {
    LOG(FATAL) << "GD_FINAL_BUF DIM NOT AGREE WITH FILTERMAP";
  }
  int count = gd_final_buf_->count();
  caffe_copy(count, gd_final_buf_->gpu_data(),
	     static_cast<Dtype*>(this->blobs_[0]->mutable_gpu_diff()));
  // Done!
}

template <typename Dtype>
void FilterMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool> &propagate_down, const vector<Blob<Dtype>*>& bottom) {


  // Step 1: Normal convolution layer gradient.
  caffe_gpu_set(main_conv_layer_->blobs_[0]->count(), Dtype(0), main_conv_layer_->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_set(main_conv_layer_->blobs_[1]->count(), Dtype(0), main_conv_layer_->blobs_[1]->mutable_gpu_diff());
  main_conv_layer_->Backward(main_conv_top_vec_, propagate_down, main_conv_bottom_vec_);

  // Step 2: backward the gradients to padded filtermap
  PostBackward();
}

#endif //IMP2

INSTANTIATE_CLASS(FilterMapLayer);
REGISTER_LAYER_CLASS(FilterMap);
}  // namespace caffe
