#pragma once

#include "colmap/controllers/global_pipeline.h"
#include "colmap/estimators/fixed_rig_global_positioning.h"

namespace colmap {

// Global reconstruction pipeline specialized for calibrated camera rigs.
// Multi-component decomposition and all shared pipeline stages remain owned by
// GlobalPipeline.
class FixedRigGlobalPipeline : public GlobalPipeline {
 public:
  FixedRigGlobalPipeline(
      GlobalPipelineOptions options,
      std::shared_ptr<Database> database,
      std::shared_ptr<ReconstructionManager> reconstruction_manager,
      FixedRigGlobalPositionerOptions positioner_options = {});
};

}  // namespace colmap
