#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigLogScalePriorScore(const float* const sensor_from_rig_log_scale,
                                const float* const target,
                                const float* const sqrt_information,
                                float* const out_rTr,
                                size_t problem_size);

}  // namespace caspar