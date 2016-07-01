#include <cstring>
#include <vector>

#include "gtest/gtest.h"


template <typename TYpeParam> class FilterMapLayerTest;

#define TEST_FRIEND \
  friend class FilterMapLayerTest<Dtype>;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/filter_map_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/filter_map_layer_impl.hpp"

namespace caffe {

template <typename TypeParam>
class FilterMapLayerTest : public MultiDeviceTest<TypeParam> {
  
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FilterMapLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 2, 10, 5)),
	blob_bottom_2_(new Blob<Dtype>(2, 2, 10, 5)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~FilterMapLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

 
typedef ::testing::Types<GPUDevice<float> > TestGPUF;
TYPED_TEST_CASE(FilterMapLayerTest, TestGPUF);



TYPED_TEST(FilterMapLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(5);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(2500);
  FilterMapParameter *fmap_param = layer_param.mutable_filter_map_param();

  fmap_param->set_map_height(25);
  fmap_param->set_map_width(100);

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  FilterMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2500);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 1);

  EXPECT_EQ(layer.blobs()[0]->shape(0), 2);
  EXPECT_EQ(layer.blobs()[0]->shape(1), 2500);
}


  // Check the FilterSampling.
TYPED_TEST(FilterMapLayerTest, TestSimpleFilterMap) {
  // A 50x50 filterMap with everything equal to 1
  typedef typename TypeParam::Dtype Dtype;

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(2);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);

  FilterMapParameter *fmap_param = layer_param.mutable_filter_map_param();

  fmap_param->set_map_height(2);
  fmap_param->set_map_width(2);

  FilterMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  caffe_set<Dtype>(layer.blobs()[0]->count(), 1, layer.blobs()[0]->mutable_cpu_data());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<Dtype> *filters = layer.main_conv_blobs()[0].get();
  DLOG(ERROR) << "Generated Conv Filter Shape: " << filters->shape_string();
  DLOG(ERROR) << "Generated Conv Filter asum: " << filters->asum_data();
  DLOG(ERROR) << "Generated Conv Filter asum_sq: " << filters->sumsq_data();
  int correct = 0;
  for (int i = 0; i < filters->count(); i++) {
    Dtype val = filters->cpu_data()[i];
    DLOG(ERROR) << val;
  }
  //EXPECT_EQ(correct, filters->count());
}

  

  
TYPED_TEST(FilterMapLayerTest, TestSimpleGrad) {
  // A 50x50 filterMap
  typedef typename TypeParam::Dtype Dtype;

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(2);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);

  FilterMapParameter *fmap_param = layer_param.mutable_filter_map_param();

  fmap_param->set_map_height(2);
  fmap_param->set_map_width(2);

  FilterMapLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.blobs()[0]->mutable_cpu_data()[0] = 0;
  layer.blobs()[0]->mutable_cpu_data()[1] = 1;
  layer.blobs()[0]->mutable_cpu_data()[2] = 2;
  layer.blobs()[0]->mutable_cpu_data()[3] = 3;

  layer.blobs()[0]->mutable_cpu_data()[4] = 4;
  layer.blobs()[0]->mutable_cpu_data()[5] = 5;
  layer.blobs()[0]->mutable_cpu_data()[6] = 6;
  layer.blobs()[0]->mutable_cpu_data()[7] = 7;

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  DLOG(ERROR) << layer.main_conv_blobs()[0]->shape_string();

  for (int i = 0; i < layer.main_conv_blobs()[0]->count(); i++) {
    DLOG(ERROR) << layer.main_conv_blobs()[0]->cpu_data()[i];
  }

  caffe_copy(layer.main_conv_blobs()[0]->count(),
	     layer.main_conv_blobs()[0]->cpu_data(),
	     layer.main_conv_blobs()[0]->mutable_cpu_diff());
	     
  layer.PostBackward();

  int correct = 0;
  for (int i = 0; i < layer.blobs()[0]->count(); i++) {
    DLOG(ERROR) << layer.blobs()[0]->cpu_diff()[i];
  }
  //EXPECT_EQ(correct, layer.blobs()[0]->count());
  //BlobProto proto;
  //layer.blobs()[2]->ToProto(&proto, true);
  //string name = "grad.proto";
  //WriteProtoToBinaryFile(proto, name);
}


TYPED_TEST(FilterMapLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  FilterMapParameter *fmap_param = layer_param.mutable_filter_map_param();

  fmap_param->set_map_height(2);
  fmap_param->set_map_width(3);
  
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(2);
  //convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  //  convolution_param->set_bias_term(false);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  FilterMapLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

  
}  // namespace caffe
