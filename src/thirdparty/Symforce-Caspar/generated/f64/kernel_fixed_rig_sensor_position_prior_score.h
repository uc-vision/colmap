#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigSensorPositionPriorScore(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const double* const sensor_from_rig_log_scale,
    double* position,
    unsigned int position_num_alloc,
    double* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    double* const out_rTr,
    size_t problem_size);

}  // namespace caspar