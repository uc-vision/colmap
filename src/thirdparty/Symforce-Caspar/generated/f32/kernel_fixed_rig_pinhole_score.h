#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPinholeScore(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          const float* const sensor_from_rig_log_scale,
                          float* calib,
                          unsigned int calib_num_alloc,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* pixel,
                          unsigned int pixel_num_alloc,
                          const float* const reprojection_loss_scale,
                          float* const out_rTr,
                          size_t problem_size);

}  // namespace caspar