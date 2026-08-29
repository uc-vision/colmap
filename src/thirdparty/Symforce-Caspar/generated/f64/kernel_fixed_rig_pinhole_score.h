#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPinholeScore(double* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          double* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          const double* const sensor_from_rig_log_scale,
                          double* calib,
                          unsigned int calib_num_alloc,
                          double* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          double* pixel,
                          unsigned int pixel_num_alloc,
                          const double* const reprojection_loss_scale,
                          double* const out_rTr,
                          size_t problem_size);

}  // namespace caspar