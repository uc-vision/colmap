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

FixedRigGlobalMapper::FixedRigGlobalMapper(
    std::shared_ptr<const DatabaseCache> database_cache,
    FixedRigGlobalPositionerOptions positioner_options)
    : mapper_(
          std::move(database_cache),
          CreateFixedRigGlobalMapperStrategy(std::move(positioner_options))) {}

void FixedRigGlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  mapper_.BeginReconstruction(reconstruction);
}

bool FixedRigGlobalMapper::Solve(const GlobalMapperOptions& options,
                                 const std::function<bool()>& on_progress) {
  return mapper_.Solve(options, on_progress);
}

bool FixedRigGlobalMapper::RotationAveraging(
    const GlobalMapperOptions& options) {
  return mapper_.RotationAveraging(options.RotationAveraging());
}

void FixedRigGlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  mapper_.EstablishTracks(options);
}

std::shared_ptr<Reconstruction> FixedRigGlobalMapper::Reconstruction() const {
  return mapper_.Reconstruction();
}

}  // namespace colmap
