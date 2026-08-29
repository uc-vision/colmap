#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigLogScalePriorScore(const double* const sensor_from_rig_log_scale,
                                const double* const target,
                                const double* const sqrt_information,
                                double* const out_rTr,
                                size_t problem_size);

}  // namespace caspar