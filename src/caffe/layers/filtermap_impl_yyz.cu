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
  template< typename T>
  __global__ void kernel_transpose_2d(T *dst, const T *src, uint Rows, uint Cols) {
    __shared__ T block[BLOCK_SIZE][BLOCK_SIZE+1];

    uint blockIdx_x, blockIdx_y;
	
#ifdef DIAGONAL_COORDINATE_TRANSPOSE
    if (Rows == Cols)
      {
	blockIdx_y = blockIdx.x;
	blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
      } else
      {
	int bid = blockIdx.x + gridDim.x*blockIdx.y;
	blockIdx_y = bid%gridDim.y;
	blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
      }
#else
    blockIdx_x = blockIdx.x; blockIdx_y = blockIdx.y;
#endif

    // read the matrix tile into shared memory
    uint xIndex = blockIdx_x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx_y * blockDim.y + threadIdx.y;
	

    if((xIndex < Cols) && (yIndex < Rows)) {
      uint index_in = yIndex * Cols + xIndex;

      block[threadIdx.y][threadIdx.x] = src[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx_y * blockDim.y + threadIdx.x;
    yIndex = blockIdx_x * blockDim.x + threadIdx.y;
	

    if((xIndex < Rows) && (yIndex < Cols)) {
      uint index_out = yIndex * Rows + xIndex;

      dst[index_out] = block[threadIdx.x][threadIdx.y];
    }
  }



  template< typename T>
  void __global__ kernel_copy_padarray_post_s1(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols && j < Rows && z < Depth)
      {
	int srcidx = z*Rows*Cols+j*Cols+i, dstidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i;

	dst[dstidx] = src[srcidx];		
      }
  }

  template< typename T>
  void __global__ kernel_copy_padarray_post_s2(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols && j < W2 && z < Depth)
      {
	int dstidx = z*(Rows+W2)*(Cols+W1)+(j+Rows)*(Cols+W1)+i;
						
	int srcidx = z*Rows*Cols+j*Cols+i;
	dst[dstidx] = src[srcidx];
			
      }
  }

  template< typename T>
  void __global__ kernel_copy_padarray_post_s3(T* dst, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < W1 && j < Rows+W2 && z < Depth)
      {
	int dstidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i;

			
	dst[dstidx+Cols] = dst[dstidx];
			
      }
  }

  template< typename T>
  __forceinline__ void compute_padarray_post(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    dim3 threadsPerBlock(16,16,4);
    dim3 blocksPerGrid( (Cols+threadsPerBlock.x-1)/threadsPerBlock.x, (Rows+threadsPerBlock.y-1)/threadsPerBlock.y, 
			(Depth+threadsPerBlock.z-1)/threadsPerBlock.z);	
    kernel_copy_padarray_post_s1<<<blocksPerGrid, threadsPerBlock>>>(dst, src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 64; threadsPerBlock.y= 4; threadsPerBlock.z= 4; 
    blocksPerGrid.x = (Cols+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (W2+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z; 
    kernel_copy_padarray_post_s2<<<blocksPerGrid, threadsPerBlock>>>(dst, src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 4; threadsPerBlock.y= 64; threadsPerBlock.z= 4;
    blocksPerGrid.x = (W1+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (Rows+W2+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z;
    kernel_copy_padarray_post_s3<<<blocksPerGrid, threadsPerBlock>>>(dst, Rows, Cols, Depth, W1, W2);
  }


  template< typename T>
  void __global__ kernel_copy_padarray_pre_s1(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols && j < Rows && z < Depth)
      {
	int srcidx = z*Rows*Cols+j*Cols+i, dstidx = z*(Rows+W2)*(Cols+W1)+(j+W2)*(Cols+W1)+i+W1;

	dst[dstidx] = src[srcidx];		
      }
  }

  template< typename T>
  void __global__ kernel_copy_padarray_pre_s2(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols && j < W2 && z < Depth)
      {
	int dstidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i+W1;
						
	int srcidx = z*Rows*Cols+(j+Rows-W2)*Cols+i;
	dst[dstidx] = src[srcidx];
			
      }
  }

  template< typename T>
  void __global__ kernel_copy_padarray_pre_s3(T* dst, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < W1 && j < Rows+W2 && z < Depth)
      {
	int dstidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i;

			
	dst[dstidx] = dst[dstidx+Cols];
			
      }
  }

  template< typename T>
  __forceinline__ void compute_padarray_pre(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    dim3 threadsPerBlock(16,16,4);
    dim3 blocksPerGrid( (Cols+threadsPerBlock.x-1)/threadsPerBlock.x, (Rows+threadsPerBlock.y-1)/threadsPerBlock.y, 
			(Depth+threadsPerBlock.z-1)/threadsPerBlock.z);	
    kernel_copy_padarray_pre_s1<<<blocksPerGrid, threadsPerBlock>>>(dst, src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 64; threadsPerBlock.y= 4; threadsPerBlock.z= 4; 
    blocksPerGrid.x = (Cols+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (W2+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z; 
    kernel_copy_padarray_pre_s2<<<blocksPerGrid, threadsPerBlock>>>(dst, src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 4; threadsPerBlock.y= 64; threadsPerBlock.z= 4;
    blocksPerGrid.x = (W1+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (Rows+W2+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z;
    kernel_copy_padarray_pre_s3<<<blocksPerGrid, threadsPerBlock>>>(dst, Rows, Cols, Depth, W1, W2);
  }

  template< typename T>
  void __global__ kernel_unpad_s1(T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols+W1 && j < W2 && z < Depth)
      {
	int srcidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i;

	src[srcidx] += src[srcidx+Rows*(Cols+W1)];		
      }
  }

  template< typename T>
  void __global__ kernel_unpad_s2(T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < W1 && j < Rows && z < Depth)
      {
	int srcidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i;
						
	src[srcidx] += src[srcidx+Cols];
			
      }
  }

  template< typename T>
  void __global__ kernel_unpad_s3(T* dst, const T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Cols && j < Rows && z < Depth)
      {
	int srcidx = z*(Rows+W2)*(Cols+W1)+j*(Cols+W1)+i, dstidx = z*Rows*Cols+j*Cols+i;

	dst[dstidx] = src[srcidx];
			
      }
  }

	
  //the function call to unpad the src.
  //src is of size Depth*(Rows+W2)*(Cols+W1)
  //dst is of size Depth*Rows*Cols
  //note that src is changed in the unpadding
  template< typename T>
  __forceinline__ void compute_unpad(T* dst, T* src, int Rows, int Cols, int Depth, int W1, int W2)
  {
    dim3 threadsPerBlock(64,4,4); 
    dim3 blocksPerGrid ( (Cols+W1+threadsPerBlock.x-1)/threadsPerBlock.x, (W2+threadsPerBlock.y-1)/threadsPerBlock.y,
			 (Depth+threadsPerBlock.z-1)/threadsPerBlock.z);

    kernel_unpad_s1<<<blocksPerGrid, threadsPerBlock>>>(src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 4; threadsPerBlock.y= 64; threadsPerBlock.z= 4;
    blocksPerGrid.x = (W1+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (Rows+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z;
    kernel_unpad_s2<<<blocksPerGrid, threadsPerBlock>>>(src, Rows, Cols, Depth, W1, W2);

    threadsPerBlock.x = 16; threadsPerBlock.y= 16; threadsPerBlock.z= 4;
    blocksPerGrid.x = (Cols+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (Rows+threadsPerBlock.y-1)/threadsPerBlock.y;
    blocksPerGrid.z = (Depth+threadsPerBlock.z-1)/threadsPerBlock.z;
    kernel_unpad_s3<<<blocksPerGrid, threadsPerBlock>>>(dst, src, Rows, Cols, Depth, W1, W2);
  }
  

  //extract filters from src, namely padded filter map
  //note that the size of padded filter map is ch*(rows+kernel_size_-1)*(cols+kernel_size_-1)
  template< typename T>
  void __global__ kernel_extract_filters(T* dst, const T* src, int rows, int cols, int ch, int kernel_size_)
  {
    __shared__ T block[4][16][16];

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int pad_rows = rows + kernel_size_-1;
    int pad_cols = cols + kernel_size_-1;
				
    int srcidx = z*pad_rows*pad_cols+j*pad_cols+i;
    int filter_idx = j*cols+i;
    int dstidx = filter_idx*ch*kernel_size_*kernel_size_ + z*kernel_size_*kernel_size_; 

    if (i < cols && j < rows && z < ch)
      {			
	block[threadIdx.z][threadIdx.y][threadIdx.x] = src[srcidx];

			
      }

    __syncthreads();			

		
    int block_x = cols - blockIdx.x * blockDim.x;
    int block_y = rows - blockIdx.y * blockDim.y;

    if (i < cols && j < rows && z < ch)
      {
			
	if (threadIdx.x <= MIN(16,block_x) - kernel_size_ && threadIdx.y <= MIN(16,block_y) - kernel_size_)
	  for (int t1 = 0; t1 < kernel_size_; t1++)
	    for (int t2 = 0; t2 < kernel_size_; t2++)
	      dst[dstidx+t1*kernel_size_+t2] = block[threadIdx.z][threadIdx.y+t1][threadIdx.x+t2];
	else
	  for (int t1 = 0; t1 < kernel_size_; t1++)
	    for (int t2 = 0; t2 < kernel_size_; t2++)
	      dst[dstidx+t1*kernel_size_+t2] = src[srcidx + t1*pad_cols+t2];
      }
			
  }
}

namespace caffe {

  template<typename Dtype>
  void FilterMapLayerImpl<Dtype>::Padding(Dtype* outdata, const Dtype *indata, int f_h, int f_w, int ch, int kernel_size) {
    filtermap::compute_padarray_post(outdata, indata, f_h , f_w , ch , kernel_size-1, kernel_size-1);
  }
 
 template<typename Dtype>
  void FilterMapLayerImpl<Dtype>::UnPadding(Dtype* outdata, Dtype *indata, int f_h, int f_w, int ch, int kernel_size) {
    filtermap::compute_unpad(outdata, indata, f_h , f_w , ch , kernel_size-1, kernel_size-1);
  }
 template<typename Dtype>
 void FilterMapLayerImpl<Dtype>::Transpose(Dtype *outdata, const Dtype *indata, int h, int w) {
   dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
   int rows = h;
   int cols = w;
   dim3 blocksPerGrid( (cols+threadsPerBlock.x-1)/threadsPerBlock.x, (rows+threadsPerBlock.y-1)/threadsPerBlock.y);
    
   filtermap::kernel_transpose_2d<<<blocksPerGrid, threadsPerBlock>>>(outdata, indata, rows, cols);

 }

  //FilterMapLayerImpl(_filterMapLayer):filterMapLayer(_filterMapLayer){}

    //extract filters from filterMapPad_weight (device pointer) to weight (device pointer)
    //f_h,f_w: height and width of the filter map
    //ch = channels_ / group_;
    //kernel_size_ kernel size

  template<typename Dtype>
  void FilterMapLayerImpl<Dtype>::ExtractFilters(Dtype* weight, Dtype* filterMapPad_weight, const Dtype* filterMap_weight, int f_h, int f_w, int ch, int kernel_size_) {
    filtermap::compute_padarray_post(filterMapPad_weight, filterMap_weight,f_h,f_w,ch,kernel_size_-1,kernel_size_-1);
    

    dim3 threadsPerBlock(16,16,4);
    dim3 blocksPerGrid( (f_w+threadsPerBlock.x-1)/threadsPerBlock.x, (f_h+threadsPerBlock.y-1)/threadsPerBlock.y, (ch+threadsPerBlock.z-1)/threadsPerBlock.z);
    
    filtermap::kernel_extract_filters<<<blocksPerGrid, threadsPerBlock>>>(weight,filterMapPad_weight,f_h,f_w,ch,kernel_size_);
  }
  
  //obtain weight_diff_tran_pad_blob and conv_sqfilter, the convolutoin of the two is the diff of the filtermap
  //weight_diff the diff of the sampled filters from the filter map dimension is (f_h*f_w)*ch*kernel_size_*kernel_size_
  //f_h,f_w: height and width of the filter map
  //ch = channels_ / group_;
  //kernel_size_ kernel size 
  template< typename Dtype>
  void FilterMapLayerImpl<Dtype>::TransPadFilterGd(shared_ptr<Blob<Dtype> > weight_diff_tran_pad_blob, shared_ptr<Blob<Dtype> > conv_sqfilter, const Dtype* weight_diff, int f_h, int f_w, int ch, int kernel_size_)
  {
    
    int weight_diff_rows = f_h*f_w;
    int weight_diff_cols = ch*kernel_size_*kernel_size_;
    
    shared_ptr<Blob<Dtype> > weight_diff_tran_blob;
    
    weight_diff_tran_blob.reset(new Blob<Dtype>(
						ch,kernel_size_*kernel_size_,f_h,f_w));

    Dtype* weight_diff_tran = weight_diff_tran_blob->mutable_gpu_data();
    
    dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocksPerGrid( (weight_diff_cols+threadsPerBlock.x-1)/threadsPerBlock.x, (weight_diff_rows+threadsPerBlock.y-1)/threadsPerBlock.y);
    
    filtermap::kernel_transpose_2d<<<blocksPerGrid, threadsPerBlock>>>(weight_diff_tran, weight_diff, weight_diff_rows, weight_diff_cols);
    
    weight_diff_tran_pad_blob->Reshape(ch,kernel_size_*kernel_size_,f_h+kernel_size_-1,f_w+kernel_size_-1);
    
    Dtype* weight_diff_tran_pad = weight_diff_tran_pad_blob->mutable_gpu_data();
    
    filtermap::compute_padarray_pre(weight_diff_tran_pad, weight_diff_tran,f_h,f_w,ch*kernel_size_*kernel_size_,kernel_size_-1,kernel_size_-1);

    
    conv_sqfilter->Reshape(1, kernel_size_*kernel_size_, kernel_size_, kernel_size_);
    
    Dtype* conv_sqfilter_weight =  conv_sqfilter->mutable_cpu_data();

    caffe_set<Dtype>(conv_sqfilter->count(), 0, conv_sqfilter_weight);
    
    for (int i = 0; i < kernel_size_; ++i)
      {
	for (int j = 0; j < kernel_size_; ++j) 
	  {
	    int ich = (kernel_size_-j-1)*kernel_size_ + kernel_size_-i-1;
	    conv_sqfilter_weight[ich*kernel_size_*kernel_size_ + j*kernel_size_ + i] = 1;
	  }
      }
    
    
    
  }
  
  /*

  template <typename Dtype>
  void FilterMapLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
				    vector<Blob<Dtype>*>* top) {
    ConvolutionLayer<Dtype>::SetUp(bottom,top);


    f_h = 50;
    f_w = 50;

    //diff
    //  this->blobs_[0].reset(new Blob<Dtype>(
    //num_output_, channels_ / group_, kernel_size_, kernel_size_));

    filterMap.reset(new Blob<Dtype>(channels_ / group_, f_h, f_w));
    filterMap_pad.reset(new Blob<Dtype>(channels_ / group_, f_h+kernel_size_-1, f_w+kernel_size_-1));

    Dtype* weight = blobs_[0]->mutable_gpu_data();
    const Dtype* filterMapPad_weight = filterMap_pad->gpu_data();

    int ch = channels_/group_;

    FilterMapLayerImpl<Dtype>::ExtractFilters(weight,filterMapPad_weight,f_h,f_w,ch,kernel_size_);
  }

  template <typename Dtype>
  Dtype FilterMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
					   vector<Blob<Dtype>*>* top) {

    ConvolutionLayer<Dtype>::Forward_gpu(bottom,top);
  }

  template <typename Dtype>
  void FilterMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
					   const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    ConvolutionLayer<Dtype>::Backward_gpu(top,propagate_down,bottom);
  }


  INSTANTIATE_CLASS(ConvolutionLayer);

  */

  template class FilterMapLayerImpl<float>;
  template class FilterMapLayerImpl<double>;

}  // namespace caffe
