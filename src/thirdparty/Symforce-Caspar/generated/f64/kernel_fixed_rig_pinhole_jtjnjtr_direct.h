#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPinholeJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    const double* const sensor_from_rig_log_scale_njtr,
    double* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar