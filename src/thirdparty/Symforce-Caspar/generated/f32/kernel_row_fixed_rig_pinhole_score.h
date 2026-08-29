#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void RowFixedRigPinholeScore(float* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             float* sensor_calibration,
                             unsigned int sensor_calibration_num_alloc,
                             SharedIndex* sensor_calibration_indices,
                             const float* const sensor_from_rig_log_scale,
                             float* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             float* pixel,
                             unsigned int pixel_num_alloc,
                             float* const out_rTr,
                             size_t problem_size);

}  // namespace caspar