#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPositionPriorScore(float* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
                                float* position,
                                unsigned int position_num_alloc,
                                float* sqrt_information,
                                unsigned int sqrt_information_num_alloc,
                                const float* const position_loss_scale,
                                float* const out_rTr,
                                size_t problem_size);

}  // namespace caspar