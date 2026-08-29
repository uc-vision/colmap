#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedCameraPinholePointScore(double* point,
                                  unsigned int point_num_alloc,
                                  SharedIndex* point_indices,
                                  double* image_from_world,
                                  unsigned int image_from_world_num_alloc,
                                  SharedIndex* image_from_world_indices,
                                  double* pixel,
                                  unsigned int pixel_num_alloc,
                                  double* const out_rTr,
                                  size_t problem_size);

}  // namespace caspar