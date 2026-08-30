#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPositionPriorScore(double* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
                                double* position,
                                unsigned int position_num_alloc,
                                double* sqrt_information,
                                unsigned int sqrt_information_num_alloc,
                                const double* const position_loss_scale,
                                double* const out_rTr,
                                size_t problem_size);

}  // namespace caspar