#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigSensorPositionPriorJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    const double* const sensor_from_rig_log_scale_njtr,
    double* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    size_t problem_size);

}  // namespace caspar