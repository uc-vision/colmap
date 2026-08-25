#include "colmap/controllers/fixed_rig_global_pipeline.h"

#include "colmap/sfm/fixed_rig_global_mapper.h"

namespace colmap {

FixedRigGlobalPipeline::FixedRigGlobalPipeline(
    GlobalPipelineOptions options,
    std::shared_ptr<Database> database,
    std::shared_ptr<ReconstructionManager> reconstruction_manager,
    FixedRigGlobalPositionerOptions positioner_options)
    : GlobalPipeline(
          std::move(options),
          std::move(database),
          std::move(reconstruction_manager),
          CreateFixedRigGlobalMapperStrategy(std::move(positioner_options))) {}

}  // namespace colmap
