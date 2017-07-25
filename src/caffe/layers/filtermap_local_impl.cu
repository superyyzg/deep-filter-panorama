// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filter_map_layer_impl.hpp"

#define BLOCK_SIZE 32

#define MIN(a,b) (((a)<(b))?(a):(b))

namespace filtermap
{

  //extract 4 3x3 filters from 3x3 mini (local) filter maps
  template< typename T>
  void __global__ kernel_extract_3x3_filters_from_3x3_fm(T* dst, const T* src, int rows, int cols, int ch, int kernel_size_)
  {
    //__shared__ T block[4][16][16];

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    
	if (i < cols && j < rows && z < ch)
    {
	    const int fm_size = 3;
		const int fm_num  = 4;
		int Row = fm_size*rows;
		int Col = fm_size*cols;
		int J = fm_size*j;
		int I = fm_size*i;
					
		int srcidx = z*Rows*Cols+J*Rows+I;
		int filter_idx = fm_num*(j*cols+i);
		int dstidx = filter_idx*ch*kernel_size_*kernel_size_ + z*kernel_size_*kernel_size_; 

		int F[9]; 
		F[0]  = src[srcidx];   F[1]  = src[srcidx+1]; F[2]  = src[srcidx+2]; 
		F[3]  = src[srcidx+3]; F[4]  = src[srcidx+4]; F[5]  = src[srcidx+5];
		F[6]  = src[srcidx+6]; F[7]  = src[srcidx+7]; F[8]  = src[srcidx+8];
		
		// the first filter
		T* f = dst[dstidx]; 
		f[0] = F[0]; f[1] = F[1];  f[2] = F[2];
		f[3] = F[3]; f[4] = F[4];  f[5] = F[5];
		f[6] = F[6]; f[7] = F[7];  f[8] = F[8];
		
		// the second filter
		f += 9;
        f[0] = F[2]; f[1] = F[1];  f[2] = F[0];
		f[3] = F[5]; f[4] = F[4];  f[5] = F[3];
		f[6] = F[8]; f[7] = F[7];  f[8] = F[6];
		
		// the third filter
		f += 9;
        f[0] = F[6]; f[1] = F[7];  f[2] = F[8];
		f[3] = F[3]; f[4] = F[4];  f[5] = F[5];
		f[6] = F[0]; f[7] = F[1];  f[8] = F[2];
		
		// the fourth filter
		f += 9;
        f[0] = F[8]; f[1] = F[7];  f[2] = F[6];          
		f[3] = F[5]; f[4] = F[4];  f[5] = F[1];
		f[6] = F[2]; f[7] = F[3];  f[8] = F[0];
	  
	}			
  
  }
  
  //extract 4 3x3 filters from 3x3 mini (local) filter maps
  template< typename T>
  void __global__ kernel_grad _aggregate_3x3_filters_to_3x3_fm(T* dst, const T* src, int rows, int cols, int ch, int kernel_size_)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    
	if (i < cols && j < rows && z < ch)
    {
	    const int fm_size = 3;
		const int fm_num  = 4;
		int Row = fm_size*rows;
		int Col = fm_size*cols;
		int J = fm_size*j;
		int I = fm_size*i;
					
        int filter_idx = fm_num*(j*cols+i);
		int srcidx = filter_idx*ch*kernel_size_*kernel_size_ + z*kernel_size_*kernel_size_;
		int dstidx = z*Rows*Cols+J*Rows+I;
		
		const T *f1 = src;
		const T *f2 = f1 + ch*kernel_size_*kernel_size_;
		const T *f3 = f2 + ch*kernel_size_*kernel_size_;
		const T *f4 = f3 + ch*kernel_size_*kernel_size_;
		
		dst[dstidx]   = f1[0] + f2[2] + f3[6] + f4[8];
		dst[dstidx+1] = f1[1] + f2[1] + f3[7] + f4[5];
		dst[dstidx+2] = f1[2] + f2[0] + f3[8] + f4[6];
		dst[dstidx+3] = f1[3] + f2[5] + f3[3] + f4[7];
		dst[dstidx+4] = f1[4] + f2[4] + f3[4] + f4[4];
		dst[dstidx+5] = f1[5] + f2[3] + f3[5] + f4[3];
		dst[dstidx+6] = f1[6] + f2[8] + f3[0] + f4[2];
		dst[dstidx+7] = f1[7] + f2[7] + f3[1] + f4[1];
		dst[dstidx+7] = f1[7] + f2[7] + f3[1] + f4[1];
	}
  }


}//namespace filtermap

namespace caffe {

  template<typename Dtype>
  void FilterMapLayerImpl<Dtype>::ExtractFilters_3x3(Dtype* weight, const Dtype* filterMap_weight, int f_h, int f_w, int ch, int kernel_size_) {
    
    dim3 threadsPerBlock(16,16,4);
    dim3 blocksPerGrid( (f_w+threadsPerBlock.x-1)/threadsPerBlock.x, (f_h+threadsPerBlock.y-1)/threadsPerBlock.y, (ch+threadsPerBlock.z-1)/threadsPerBlock.z);
	
	filtermap::kernel_extract_3x3_filters_from_3x3_fm<<<blocksPerGrid, threadsPerBlock>>>(weight, filterMap_weight, f_h,f_w,ch,kernel_size_);
    
  }
  
  template< typename Dtype>
  void FilterMapLayerImpl<Dtype>::AggregateGradForFilterMap(shared_ptr<Blob<Dtype> > fm_gd_blob, const Dtype* weight_diff, int f_h, int f_w, int ch, int kernel_size_)
  {
	dim3 threadsPerBlock(16,16,4);
    dim3 blocksPerGrid( (f_w+threadsPerBlock.x-1)/threadsPerBlock.x, (f_h+threadsPerBlock.y-1)/threadsPerBlock.y, (ch+threadsPerBlock.z-1)/threadsPerBlock.z);
	
	filtermap::kernel_grad _aggregate_3x3_filters_to_3x3_fm<<<blocksPerGrid, threadsPerBlock>>>(fm_gd_blob->mutable_gpu_data(), weight_diff, f_h, f_w, ch, kernel_size_);
  }
  
}