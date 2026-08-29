#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigLogScalePriorJtjnjtrDirect(
    const double* const sensor_from_rig_log_scale_njtr,
    double* sensor_from_rig_log_scale_jac,
    unsigned int sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    size_t problem_size);

}  // namespace caspar