#pragma once

#include "colmap/estimators/global_positioning.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

namespace colmap {

struct FixedRigGlobalPositionerOptions {
  // Seed frame positions from camera pose priors.
  bool initialize_from_pose_priors = false;
};

// Metric global positioner for a calibrated camera rig. The known sensor
// baselines eliminate the independent per-observation scales used by BATA.
class FixedRigGlobalPositioner {
 public:
  FixedRigGlobalPositioner(const GlobalPositionerOptions& options,
                           const FixedRigGlobalPositionerOptions& rig_options,
                           double min_tri_angle_deg);

  bool Solve(const PoseGraph& pose_graph,
             Reconstruction& reconstruction,
             const std::vector<PosePrior>& pose_priors);

 private:
  struct FrameConstraint {
    frame_t frame_id1;
    frame_t frame_id2;
    Eigen::Vector3d center2_from_center1;
    double weight;
  };

  void SetupProblem();
  FlatHashSet<frame_t> ConstrainedFrameIds(
      const PoseGraph& pose_graph, const Reconstruction& reconstruction) const;
  void InitializePositions(const PoseGraph& pose_graph,
                           Reconstruction& reconstruction,
                           const std::vector<PosePrior>& pose_priors);
  bool InitializePositionsFromPosePriors(
      const PoseGraph& pose_graph,
      const Reconstruction& reconstruction,
      const std::vector<PosePrior>& pose_priors,
      const FlatHashSet<frame_t>& constrained_frame_ids);
  void AddPointConstraints(Reconstruction& reconstruction);
  void AddPointConstraint(point3D_t point3D_id, Reconstruction& reconstruction);
  void AddParameterBlockOrdering(Reconstruction& reconstruction);
  void ParameterizeVariables(Reconstruction& reconstruction);
  void ConfigureSolver(const Reconstruction& reconstruction);
  void ConvertBackResults(Reconstruction& reconstruction);
  void RefinePoints(Reconstruction& reconstruction) const;
  bool BuildFrameConstraints(const Reconstruction& reconstruction,
                             std::vector<FrameConstraint>& constraints) const;
  bool SolveFramePositions(Reconstruction& reconstruction,
                           const std::vector<FrameConstraint>& constraints);

  GlobalPositionerOptions options_;
  FixedRigGlobalPositionerOptions rig_options_;
  double min_tri_angle_deg_;
  bool initialized_from_pose_priors_ = false;
  std::unique_ptr<ceres::Problem> problem_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> calibrated_loss_function_;
  std::shared_ptr<ceres::LossFunction> uncalibrated_loss_function_;
  NodeHashMap<frame_t, Eigen::Vector3d> frame_centers_;
};

bool RunFixedRigGlobalPositioning(
    const GlobalPositionerOptions& options,
    const FixedRigGlobalPositionerOptions& rig_options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    double min_tri_angle_deg);

}  // namespace colmap
