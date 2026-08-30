#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPositionPriorResJac(float* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 float* position,
                                 unsigned int position_num_alloc,
                                 float* sqrt_information,
                                 unsigned int sqrt_information_num_alloc,
                                 const float* const position_loss_scale,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
                                 float* const out_pose_njtr,
                                 unsigned int out_pose_njtr_num_alloc,
                                 float* const out_pose_precond_diag,
                                 unsigned int out_pose_precond_diag_num_alloc,
                                 float* const out_pose_precond_tril,
                                 unsigned int out_pose_precond_tril_num_alloc,
                                 size_t problem_size);

}  // namespace caspar