#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dot_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::ifstream;

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class DotProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DotProductLayerTest() : blob_top_(new Blob<Dtype>()) {
    // fill the values
    vector<int> shape_b(3);
    shape_b[0] = 2;
    shape_b[1] = 4;
    shape_b[2] = 5;
    vector<int> shape_a(2);
    shape_a[0] = 2;
    shape_a[1] = 4;

    blob_bottom_b_ = new Blob<Dtype>(shape_b);
    blob_bottom_a_ = new Blob<Dtype>(shape_a);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DotProductLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype> *blob_bottom_a_;
  Blob<Dtype> *blob_bottom_b_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_CASE(DotProductLayerTest, TestDtypesAndDevices);
// TYPED_TEST_CASE(DotProductLayerTest, TestCPUF);
TYPED_TEST(DotProductLayerTest, TestSetUp_VM) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_a_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_b_);
  LayerParameter layer_param;
  DotProductParameter *dot_product_param =
      layer_param.mutable_dot_product_param();
  dot_product_param->set_mode(DotProductParameter_Mode_VM);
  shared_ptr<DotProductLayer<Dtype> > layer(
      new DotProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
}

TYPED_TEST(DotProductLayerTest, TestForward_VM) {
  typedef typename TypeParam::Dtype Dtype;
  BlobProto proto_a, proto_b, proto_c;
  ifstream a_in("src/caffe/test/test_data/dot_prodoct_a_vm.binaryproto");
  proto_a.ParseFromIstream(&a_in);
  a_in.close();
  this->blob_bottom_a_->FromProto(proto_a);

  ifstream b_in("src/caffe/test/test_data/dot_prodoct_b_vm.binaryproto");
  proto_b.ParseFromIstream(&b_in);
  b_in.close();
  this->blob_bottom_b_->FromProto(proto_b);

  ifstream c_in("src/caffe/test/test_data/dot_prodoct_c_vm.binaryproto");
  proto_c.ParseFromIstream(&c_in);
  c_in.close();
  Blob<Dtype> reference_c;
  reference_c.FromProto(proto_c);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_a_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_b_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU || sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    DotProductParameter *dot_product_param =
        layer_param.mutable_dot_product_param();
    dot_product_param->set_mode(DotProductParameter_Mode_VM);

    shared_ptr<DotProductLayer<Dtype> > layer(
        new DotProductLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype *data = this->blob_top_->cpu_data();
    const Dtype *ref_data = reference_c.cpu_data();
    const int count = this->blob_top_->count();

    Dtype sum_sq_diff = 0.0;
    for (int i = 0; i < count; ++i) {
      Dtype diff = data[i] - ref_data[i];
      sum_sq_diff += diff * diff;
    }
    EXPECT_LE(sum_sq_diff, 0.00001);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}



/*
TYPED_TEST(InnerProductLayerTest, TestForwardNoBatch) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
*/

TYPED_TEST(DotProductLayerTest, TestGradient_VM) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_a_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_b_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU || sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    DotProductParameter *dot_product_param =
        layer_param.mutable_dot_product_param();
    dot_product_param->set_mode(DotProductParameter_Mode_VM);

    DotProductLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
