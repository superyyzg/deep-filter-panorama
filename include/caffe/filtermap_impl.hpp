#ifndef FILTERMAP_LAYER_HPP_
#define FILTERMAP_LAYER_HPP_


template <typename Dtype>
void transpose_gpu(Dtype *odata, const Dtype *idata, int width, int height);

template <typename Dtype>
void map_padding_gpu(const Dtype * map, Dtype *padded_map, int ch, int map_size, int pad_size);

template <typename Dtype>
void map_unpadding_gpu(const Dtype * padded_map, Dtype *map, int ch, int map_size, int pad_size);

#endif //FILTERMAP_LAYER_HPP_
