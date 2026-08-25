#include "colmap/sfm/fixed_rig_global_mapper.h"

#include "colmap/estimators/fixed_rig_global_positioning.h"
#include "colmap/estimators/fixed_rig_rotation_averaging.h"

namespace colmap {
namespace {

class FixedRigGlobalMapperStrategy final : public GlobalMapperStrategy {
 public:
  explicit FixedRigGlobalMapperStrategy(
      FixedRigGlobalPositionerOptions positioner_options)
      : positioner_options_(std::move(positioner_options)) {}

  GlobalMapperOptions Configure(
      const GlobalMapperOptions& options) const override {
    GlobalMapperOptions configured_options = options;
    configured_options.refine_sensor_from_rig = false;
    BundleAdjustmentOptions& bundle_adjustment =
        configured_options.bundle_adjustment;
    if (bundle_adjustment.backend ==
        BundleAdjustmentBackend::CASPAR_RIG_SCHUR) {
      configured_options.ba_skip_fixed_rotation_stage = true;
      configured_options.ba_skip_joint_optimization_stage = false;
      bundle_adjustment.refine_focal_length = false;
      bundle_adjustment.refine_principal_point = false;
      bundle_adjustment.refine_extra_params = false;
      bundle_adjustment.refine_rig_from_world = true;
      bundle_adjustment.refine_points3D = true;
      bundle_adjustment.constant_rig_from_world_rotation = false;
    }
    return configured_options;
  }

  void PrepareRotationAveraging(
      PoseGraph& pose_graph,
      const Reconstruction& reconstruction,
      const RotationEstimatorOptions& options) const override {
    FilterFixedRigRotationOutliers(
        pose_graph, reconstruction, options.max_rotation_error_deg);
  }

  bool RunPositioning(const GlobalPositionerOptions& options,
                      const PoseGraph& pose_graph,
                      Reconstruction& reconstruction,
                      const std::vector<PosePrior>& pose_priors,
                      double min_tri_angle_deg) const override {
    return RunFixedRigGlobalPositioning(options,
                                        positioner_options_,
                                        pose_graph,
                                        reconstruction,
                                        pose_priors,
                                        min_tri_angle_deg);
  }

 private:
  FixedRigGlobalPositionerOptions positioner_options_;
};

}  // namespace

std::shared_ptr<const GlobalMapperStrategy> CreateFixedRigGlobalMapperStrategy(
    FixedRigGlobalPositionerOptions positioner_options) {
  return std::make_shared<FixedRigGlobalMapperStrategy>(
      std::move(positioner_options));
}

}  // namespace colmap
