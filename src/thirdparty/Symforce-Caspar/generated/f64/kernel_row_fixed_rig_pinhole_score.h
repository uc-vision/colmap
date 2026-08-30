#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void RowFixedRigPinholeScore(double* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             double* sensor_calibration,
                             unsigned int sensor_calibration_num_alloc,
                             SharedIndex* sensor_calibration_indices,
                             const double* const sensor_from_rig_log_scale,
                             double* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             double* pixel,
                             unsigned int pixel_num_alloc,
                             const double* const reprojection_loss_scale,
                             double* const out_rTr,
                             size_t problem_size);

}  // namespace caspar