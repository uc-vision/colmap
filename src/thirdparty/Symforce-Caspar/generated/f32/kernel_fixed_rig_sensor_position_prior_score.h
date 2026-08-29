#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigSensorPositionPriorScore(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* position,
    unsigned int position_num_alloc,
    float* sqrt_information,
    unsigned int sqrt_information_num_alloc,
    float* const out_rTr,
    size_t problem_size);

}  // namespace caspar