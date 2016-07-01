#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/filter_map_layer_impl.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {



template<typename Dtype>
void ref_unpadding(const Blob<Dtype>* in, Blob<Dtype>* out, int padding_size) {
  CHECK_EQ(3, in->num_axes());
  // ch*fhp*fwp

  vector<int> upshape(3);
  upshape[0] = in->shape(0);
  upshape[1] = in->shape(1) - padding_size;
  upshape[2] = in->shape(2) - padding_size;

  
  out->Reshape(upshape);
  for (int ch = 0; ch < upshape[0]; ++ch) {
    for (int i = 0; i < upshape[1]; ++i) {
      for (int j = 0; j < upshape[2]; ++j) {
	out->mutable_cpu_data()[out->offset(ch,i,j)] = in->cpu_data()[in->offset(ch,i,j)];
      }
    }

    for (int i = 0; i < padding_size; ++i)
      for (int j = 0; j < upshape[2]; ++j) {
	out->mutable_cpu_data()[out->offset(ch,i,j)] += in->cpu_data()[in->offset(ch,i+upshape[1],j)];
      }

    for (int i = 0; i < upshape[1]; ++i)
      for (int j = 0; j < padding_size; ++j) {
	out->mutable_cpu_data()[out->offset(ch,i,j)] += in->cpu_data()[in->offset(ch,i,j+upshape[2])];
      }
    for (int i = 0; i < padding_size; ++i)
      for (int j = 0; j < padding_size; ++j) {
	out->mutable_cpu_data()[out->offset(ch,i,j)] += in->cpu_data()[in->offset(ch,i+upshape[1],j+upshape[2])];
      }
  }
}

template<typename Dtype>
void impl_unpadding(Blob<Dtype>* in, Blob<Dtype>* out, int padding_size) {
  CHECK_EQ(3, in->num_axes());
  // ch*fhp*fwp

  vector<int> upshape(3);
  upshape[0] = in->shape(0);
  upshape[1] = in->shape(1) - padding_size;
  upshape[2] = in->shape(2) - padding_size;

  
  out->Reshape(upshape);
  FilterMapLayerImpl<Dtype>::UnPadding(out->mutable_gpu_data(), in->mutable_gpu_data(),
				       upshape[1], upshape[2], upshape[0], padding_size+1);
}


template<typename Dtype>
void ref_trans(const Blob<Dtype>* in, Blob<Dtype>* out, int trans_dim1, bool diff=false) {
  out->ReshapeLike(*in);
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

template<typename Dtype>
void impl_trans(const Blob<Dtype>* in, Blob<Dtype>* out, int trans_dim1) {
  out->ReshapeLike(*in);
  int trans_dim2 = in->count() / trans_dim1;
  FilterMapLayerImpl<Dtype>::Transpose(out->mutable_gpu_data(), in->gpu_data(), trans_dim1, trans_dim2);
}

template <typename TypeParam>
class FilterMapImplTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FilterMapImplTest() {}
  virtual void SetUp() {
  }
  virtual ~FilterMapImplTest() {
  }
};


typedef ::testing::Types<GPUDevice<float> > TestGPUF;

TYPED_TEST_CASE(FilterMapImplTest, TestGPUF);

TYPED_TEST(FilterMapImplTest, TestTranspose) {

  typedef typename TypeParam::Dtype Dtype;
  vector<int> dim1, dim2;
  dim1.push_back(4);  dim2.push_back(5);
  dim1.push_back(5);  dim2.push_back(4);
  dim1.push_back(40);  dim2.push_back(50);
  dim1.push_back(50);  dim2.push_back(40);
  dim1.push_back(6);  dim2.push_back(20);

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int ishape = 0; ishape < dim1.size(); ++ishape) {
    vector<int> testshape(2);
    testshape[0] = dim1[ishape];
    testshape[1] = dim2[ishape];
    Blob<Dtype> testBlob(testshape), ref_out, impl_out, diff(testshape);
    filler.Fill(&testBlob);
    impl_trans(&testBlob, &impl_out, dim1[ishape]);
    ref_trans(&testBlob, &ref_out, dim1[ishape]);
    caffe_sub(
	      ref_out.count(),
	      ref_out.cpu_data(),
	      impl_out.cpu_data(),
	      diff.mutable_cpu_data());
    EXPECT_EQ(diff.asum_data(), 0);
  }
}

TYPED_TEST(FilterMapImplTest, TestUnpaddingSimple) {

  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  vector<int> testshape(3);
  testshape[0] = 3;
  testshape[1] = 5;
  testshape[2] = 5;
  Blob<Dtype> testBlob(testshape), ref_out, impl_out, diff;
  filler.Fill(&testBlob);
  ref_unpadding(&testBlob, &ref_out, 1);
  impl_unpadding(&testBlob, &impl_out, 1);
  diff.ReshapeLike(ref_out);
      caffe_sub(
		ref_out.count(),
		ref_out.cpu_data(),
		impl_out.cpu_data(),
		diff.mutable_cpu_data());
      EXPECT_EQ(diff.asum_data(), 0);
}

TYPED_TEST(FilterMapImplTest, TestUnpadding) {

  typedef typename TypeParam::Dtype Dtype;
  vector<int> dim1, dim2;
  dim1.push_back(40);  dim2.push_back(50);
  dim1.push_back(50);  dim2.push_back(40);

  vector<int> padding_size;
  padding_size.push_back(1);
  padding_size.push_back(2);
  padding_size.push_back(3);
  padding_size.push_back(4);
  padding_size.push_back(5);
  
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int ishape = 0; ishape < dim1.size(); ++ishape) {
    for (int ips = 0; ips < padding_size.size(); ++ips) {
      vector<int> testshape(3);
      testshape[0] = 3;
      testshape[1] = dim1[ishape];
      testshape[2] = dim2[ishape];
      Blob<Dtype> testBlob(testshape), ref_out, impl_out, diff;
      filler.Fill(&testBlob);
      ref_unpadding(&testBlob, &ref_out, padding_size[ips]);
      impl_unpadding(&testBlob, &impl_out, padding_size[ips]);
      diff.ReshapeLike(ref_out);
      caffe_sub(
		ref_out.count(),
		ref_out.cpu_data(),
		impl_out.cpu_data(),
		diff.mutable_cpu_data());
      EXPECT_NEAR(diff.asum_data(), 0, 1e-7*diff.count());
    }
  }
}

  
}  // namespace caffe
