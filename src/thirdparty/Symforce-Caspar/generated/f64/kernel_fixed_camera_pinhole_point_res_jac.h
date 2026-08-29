#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedCameraPinholePointResJac(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar