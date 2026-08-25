#pragma once

#include "colmap/estimators/fixed_rig_global_positioning.h"
#include "colmap/sfm/global_mapper.h"

namespace colmap {

std::shared_ptr<const GlobalMapperStrategy> CreateFixedRigGlobalMapperStrategy(
    FixedRigGlobalPositionerOptions positioner_options = {});

}  // namespace colmap
