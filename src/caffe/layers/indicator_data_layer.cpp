#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/indicator_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
IndicatorDataLayer<Dtype>::~IndicatorDataLayer<Dtype>() {
}

template <typename Dtype>
void IndicatorDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read the file with filenames and labels
  const string& source = this->layer_param_.indicator_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;

  int dim1, dim2, num_pairs;

  infile >> dim1 >> dim2 >> num_pairs;

  idx1.resize(num_pairs);
  idx2.resize(num_pairs);
  score.resize(num_pairs);

  data_permutation_.resize(num_pairs);


  for (int i = 0; i < num_pairs; i++) {
    data_permutation_[i] = i;
  }

  for (int i = 0; i < num_pairs; i++) {
    infile >> idx1[i] >> idx2[i] >> score[i];
    idx1[i] -= 1;
    idx2[i] -= 1;
  }
  
  if (this->layer_param_.indicator_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleEntries();
  }
  LOG(INFO) << "A total of " << idx1.size() << " entries.";

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.indicator_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  top[0]->Reshape(batch_size, dim1, 1, 1);
  top[1]->Reshape(batch_size, dim2, 1, 1);
  // label
  vector<int> label_shape(1, batch_size);
  top[2]->Reshape(label_shape);
  current_row = 0;
}

template <typename Dtype>
void IndicatorDataLayer<Dtype>::ShuffleEntries() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  shuffle(data_permutation_.begin(), data_permutation_.end(), prefetch_rng);
}

template <typename Dtype>
void IndicatorDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
  Dtype* data = top[0]->mutable_cpu_data();
  for (int i = 0; i < top[0]->count(); i++) {
    data[i] = 0;
  }
  data = top[1]->mutable_cpu_data();
  for (int i = 0; i < top[1]->count(); i++) {
    data[i] = 0;
  }

  int batch_size = top[0]->num();
  for (int i = 0; i < batch_size; i++) {
    int tp = data_permutation_[current_row];
    int tidx1 = idx1[tp];
    int tidx2 = idx2[tp];

    float tscore = score[tp];

    top[0]->mutable_cpu_data()[top[0]->offset(i, tidx1)] = 1;
    top[1]->mutable_cpu_data()[top[1]->offset(i, tidx2)] = 1;
    top[2]->mutable_cpu_data()[i] = tscore;
      
    current_row++;
    if (current_row >= data_permutation_.size()) {
      current_row = 0;
      if (this->layer_param_.indicator_data_param().shuffle()) {
	ShuffleEntries();
      }
    }
  }
}
  
INSTANTIATE_CLASS(IndicatorDataLayer);
REGISTER_LAYER_CLASS(IndicatorData);

}  // namespace caffe
