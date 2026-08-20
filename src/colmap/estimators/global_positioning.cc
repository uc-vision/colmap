#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

struct FixedRigFramePositionCostFunctor {
  explicit FixedRigFramePositionCostFunctor(
      const Eigen::Vector3d& center2_from_center1, double weight)
      : center2_from_center1_(center2_from_center1), weight_(weight) {}

  template <typename T>
  bool operator()(const T* center1, const T* center2, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vector(residuals);
    residuals_vector =
        T(weight_) * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center2) -
                      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center1) -
                      center2_from_center1_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& center2_from_center1, double weight) {
    return new ceres::
        AutoDiffCostFunction<FixedRigFramePositionCostFunctor, 3, 3, 3>(
            new FixedRigFramePositionCostFunctor(center2_from_center1, weight));
  }

  const Eigen::Vector3d center2_from_center1_;
  const double weight_;
};

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options,
                                   double min_tri_angle_deg)
    : options_(options), min_tri_angle_deg_(min_tri_angle_deg) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction) {
  return Solve(pose_graph, reconstruction, {});
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction,
                             const std::vector<PosePrior>& pose_priors) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  LOG(INFO) << "Setting up the global positioner problem";
  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);

  Timer stage_timer;
  stage_timer.Start();

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction, pose_priors);
  LOG(INFO) << "Global positioner initialization done in "
            << stage_timer.ElapsedSeconds() << " seconds";

  if (fixed_rig_positioning_ && options_.optimize_positions &&
      options_.optimize_points) {
    std::vector<FixedRigFrameConstraint> constraints;
    stage_timer.Restart();
    if (BuildFixedRigFrameConstraints(reconstruction, constraints)) {
      LOG(INFO) << "Fixed-rig frame constraints built in "
                << stage_timer.ElapsedSeconds() << " seconds ("
                << constraints.size() << " constraints)";
      return SolveFixedRigFramePositions(reconstruction, constraints);
    }
    LOG(INFO)
        << "Fixed-rig frame constraints did not cover the reconstruction; "
           "using observation-level positioning";
  }

  if (pose_prior_initialization_) {
    stage_timer.Restart();
    ConvertBackResults(reconstruction);
    RefineFixedRigPoints(reconstruction);
    LOG(INFO) << "Initial fixed-rig point refinement done in "
              << stage_timer.ElapsedSeconds() << " seconds";
  }

  // Add the point to camera constraints to the problem.
  stage_timer.Restart();
  AddPointToCameraConstraints(reconstruction);
  LOG(INFO) << "Global positioner residual construction done in "
            << stage_timer.ElapsedSeconds() << " seconds ("
            << problem_->NumResidualBlocks() << " residuals, "
            << problem_->NumParameterBlocks() << " parameter blocks, "
            << frame_centers_.size() << " frames, "
            << options_.solver_options.num_threads << " threads)";

  stage_timer.Restart();
  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);
  LOG(INFO) << "Global positioner parameterization done in "
            << stage_timer.ElapsedSeconds() << " seconds";

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  stage_timer.Restart();
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  LOG(INFO) << "Global positioner Ceres solve done in "
            << stage_timer.ElapsedSeconds() << " seconds";

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  stage_timer.Restart();
  ConvertBackResults(reconstruction);
  LOG(INFO) << "Global positioner result conversion done in "
            << stage_timer.ElapsedSeconds() << " seconds";
  if (fixed_rig_positioning_) {
    stage_timer.Restart();
    RefineFixedRigPoints(reconstruction);
    LOG(INFO) << "Final fixed-rig point refinement done in "
              << stage_timer.ElapsedSeconds() << " seconds";
  }
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();

  fixed_rig_positioning_ =
      !options_.refine_sensor_from_rig &&
      std::any_of(reconstruction.Images().begin(),
                  reconstruction.Images().end(),
                  [](const auto& image_entry) {
                    return !image_entry.second.IsRefInFrame();
                  });

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid tracks, the actual number of residuals may be
  // smaller.
  scales_.clear();
  if (!fixed_rig_positioning_) {
    size_t total_observations = 0;
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      total_observations += point3D.track.Length();
    }
    scales_.reserve(total_observations);
  }
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) {
  std::unordered_set<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  pose_prior_initialization_ =
      options_.initialize_from_pose_priors && fixed_rig_positioning_ &&
      InitializePositionsFromPosePriors(
          pose_graph, reconstruction, pose_priors, constrained_positions);

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    if (pose_prior_initialization_) {
      continue;
    } else if (options_.generate_random_positions &&
               options_.optimize_positions) {
      frame_centers_[frame_id] =
          (fixed_rig_positioning_ ? 1.0 : 100.0) * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

bool GlobalPositioner::InitializePositionsFromPosePriors(
    const PoseGraph& pose_graph,
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    const std::unordered_set<frame_t>& constrained_positions) {
  std::unordered_map<image_t, Eigen::Vector3d> image_positions;
  std::unordered_map<frame_t, Eigen::Vector3d> frame_positions;
  for (const PosePrior& prior : pose_priors) {
    if (!prior.HasPosition() ||
        prior.corr_data_id.sensor_id.type != SensorType::CAMERA ||
        !reconstruction.ExistsImage(prior.corr_data_id.id)) {
      continue;
    }
    const image_t image_id = prior.corr_data_id.id;
    const Image& image = reconstruction.Image(image_id);
    const frame_t frame_id = image.FrameId();
    image_positions.emplace(image_id, prior.position);
    if (image.IsRefInFrame() &&
        constrained_positions.find(frame_id) != constrained_positions.end()) {
      frame_positions.emplace(frame_id, prior.position);
    }
  }
  if (frame_positions.size() < constrained_positions.size()) {
    return false;
  }

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto prior1 = image_positions.find(image_id1);
    const auto prior2 = image_positions.find(image_id2);
    if (prior1 == image_positions.end() || prior2 == image_positions.end()) {
      continue;
    }
    const Eigen::Vector3d prior_direction = prior2->second - prior1->second;
    if (prior_direction.squaredNorm() == 0.0) {
      continue;
    }
    const Eigen::Vector3d world_direction =
        reconstruction.Image(image_id1).CamFromWorld().rotation().inverse() *
        edge.cam2_from_cam1.TgtOriginInSrc();
    covariance += edge.num_matches * world_direction.normalized() *
                  prior_direction.normalized().transpose();
  }
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d sign = Eigen::Matrix3d::Identity();
  sign(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  const Eigen::Matrix3d world_from_prior =
      svd.matrixU() * sign * svd.matrixV().transpose();

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const frame_t frame_id : constrained_positions) {
    mean += frame_positions.at(frame_id);
  }
  mean /= constrained_positions.size();
  double squared_scale = 0.0;
  for (const frame_t frame_id : constrained_positions) {
    squared_scale += (frame_positions.at(frame_id) - mean).squaredNorm();
  }
  const double scale =
      std::sqrt(squared_scale / constrained_positions.size());
  for (const frame_t frame_id : constrained_positions) {
    frame_centers_[frame_id] =
        world_from_prior * (frame_positions.at(frame_id) - mean) / scale;
  }
  LOG(INFO) << "Initialized " << constrained_positions.size()
            << " fixed-rig frame positions from pose priors";
  return true;
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
      loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  loss_function_ptcam_calibrated_ = loss_function_;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction);
  }
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization && !pose_prior_initialization_) {
    point3D.xyz =
        (fixed_rig_positioning_ ? 1.0 : 100.0) * RandVector3d(-1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : point3D.track.Elements()) {
    if (!reconstruction.ExistsImage(observation.image_id)) continue;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) continue;

    const std::optional<Eigen::Vector3d> cam_ray =
        image.CameraPtr()->CamRayFromImg(
            image.Point2D(observation.point2D_idx).xy);
    if (!cam_ray.has_value()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to project: point3D_id="
          << point3D_id << ", image_id=" << observation.image_id
          << ", feature_id=" << observation.point2D_idx;
      continue;
    }

    const Eigen::Vector3d cam_from_point3D_dir =
        image.CamFromWorld().rotation().inverse() * (*cam_ray);

    // Down-weight uncalibrated cameras.
    Camera& camera = reconstruction.Camera(image.CameraId());
    ceres::LossFunction* loss_function =
        camera.has_prior_focal_length
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    Eigen::Vector3d cam_from_rig_dir = Eigen::Vector3d::Zero();
    if (!image.IsRefInFrame()) {
      const Rig& rig = reconstruction.Rig(image.FramePtr()->RigId());
      const Rigid3d& cam_from_rig =
          rig.SensorFromRig(image.CameraPtr()->SensorId());
      if (fixed_rig_positioning_) {
        THROW_CHECK(!cam_from_rig.translation().hasNaN())
            << "Fixed-rig positioning requires known sensor_from_rig";
        cam_from_rig_dir = image.CamFromWorld().rotation().inverse() *
                           cam_from_rig.translation();
      } else if (!cam_from_rig.translation().hasNaN()) {
        cam_from_rig_dir = image.CamFromWorld().rotation().inverse() *
                           cam_from_rig.translation();
      }
    }

    if (fixed_rig_positioning_) {
      problem_->AddResidualBlock(
          FixedRigPairwiseDirectionCostFunctor::Create(
              cam_from_point3D_dir, cam_from_rig_dir),
          loss_function,
          point3D.xyz.data(),
          frame_centers_[image.FrameId()].data());
      continue;
    }

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    if (!options_.generate_scales && random_initialization) {
      const Eigen::Vector3d cam_from_point3D_translation =
          point3D.xyz - frame_centers_[image.FrameId()];
      scale = std::max(1e-5,
                       cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                           cam_from_point3D_translation.squaredNorm());
    }

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.IsRefInFrame()) {
      ceres::CostFunction* cost_function =
          BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 frame_centers_[image.FrameId()].data(),
                                 point3D.xyz.data(),
                                 &scale);
    } else {
      // If the image is part of a camera rig, use the RigBATA error.

      const rig_t rig_id = image.FramePtr()->RigId();
      Rig& rig = reconstruction.Rig(rig_id);
      Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

      if (!cam_from_rig.translation().hasNaN()) {
        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
                cam_from_point3D_dir, cam_from_rig_dir);

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   &scale);
      } else {
        // NaN translation means the sensor's cam_from_rig must be
        // re-estimated, which requires refine_sensor_from_rig=true.
        THROW_CHECK(options_.refine_sensor_from_rig)
            << "sensor_from_rig has NaN translation but "
               "refine_sensor_from_rig=false (image_id="
            << observation.image_id << ")";
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
          // Will be initialized to random values in ParameterizeVariables().
          cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
        }

        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionCostFunctor::Create(
                cam_from_point3D_dir,
                image.FramePtr()->RigFromWorld().rotation());

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   cams_in_rig_[sensor_id].data(),
                                   &scale);
      }
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (reconstruction.NumPoints3D() > 0) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }
}

void GlobalPositioner::RefineFixedRigPoints(Reconstruction& reconstruction) {
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());

  std::vector<Eigen::Vector3d> refined_points(point3D_ids.size());
  const int num_threads = options_.solver_options.num_threads;
  const size_t chunk_size =
      (point3D_ids.size() + num_threads - 1) / num_threads;
  const Reconstruction& source = reconstruction;
  ThreadPool thread_pool(num_threads);
  for (size_t chunk_begin = 0; chunk_begin < point3D_ids.size();
       chunk_begin += chunk_size) {
    const size_t chunk_end =
        std::min(chunk_begin + chunk_size, point3D_ids.size());
    thread_pool.AddTask([&, chunk_begin, chunk_end]() {
      for (size_t point_index = chunk_begin; point_index < chunk_end;
           ++point_index) {
        const Point3D& point3D = source.Point3D(point3D_ids[point_index]);

        Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
        Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
        for (const TrackElement& observation : point3D.track.Elements()) {
          const Image& image = source.Image(observation.image_id);
          const Eigen::Vector3d cam_ray =
              image.CameraPtr()
                  ->CamRayFromImg(image.Point2D(observation.point2D_idx).xy)
                  .value();
          const Eigen::Vector3d world_ray =
              image.CamFromWorld().rotation().inverse() * cam_ray;
          const Eigen::Matrix3d ray_projection =
              Eigen::Matrix3d::Identity() - world_ray * world_ray.transpose();
          normal_matrix += ray_projection;
          right_hand_side += ray_projection * image.ProjectionCenter();
        }
        refined_points[point_index] =
            normal_matrix.ldlt().solve(right_hand_side);
      }
    });
  }
  thread_pool.Wait();

  for (size_t point_index = 0; point_index < point3D_ids.size();
       ++point_index) {
    reconstruction.Point3D(point3D_ids[point_index]).xyz =
        refined_points[point_index];
  }
}

bool GlobalPositioner::BuildFixedRigFrameConstraints(
    const Reconstruction& reconstruction,
    std::vector<FixedRigFrameConstraint>& constraints) const {
  struct RigObservation {
    frame_t frame_id;
    image_t image_id;
    point2D_t point2D_idx;
    Eigen::Vector3d center;
    Eigen::Vector3d bearing;
  };
  struct RigPoint {
    frame_t frame_id;
    Eigen::Vector3d xyz;
  };

  std::unordered_map<frame_t, std::vector<frame_t>> adjacency;
  adjacency.reserve(frame_centers_.size());
  const double minimum_cosine = std::cos(DegToRad(min_tri_angle_deg_));
  constraints.clear();
  constraints.reserve(reconstruction.NumPoints3D());

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    std::vector<RigObservation> observations;
    observations.reserve(point3D.track.Length());
    for (const TrackElement& track_element : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(track_element.image_id);
      const Eigen::Vector3d cam_ray =
          image.CameraPtr()
              ->CamRayFromImg(image.Point2D(track_element.point2D_idx).xy)
              .value();
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      Eigen::Vector3d bearing = cam_ray;
      if (!image.IsRefInFrame()) {
        const Rig& rig = reconstruction.Rig(image.FramePtr()->RigId());
        const Rigid3d& cam_from_rig =
            rig.SensorFromRig(image.CameraPtr()->SensorId());
        center = cam_from_rig.TgtOriginInSrc();
        bearing = cam_from_rig.rotation().inverse() * cam_ray;
      }
      observations.push_back(RigObservation{image.FrameId(),
                                            image.ImageId(),
                                            track_element.point2D_idx,
                                            center,
                                            bearing});
    }
    std::sort(observations.begin(),
              observations.end(),
              [](const RigObservation& observation1,
                 const RigObservation& observation2) {
                return std::tie(observation1.frame_id,
                                observation1.image_id,
                                observation1.point2D_idx) <
                       std::tie(observation2.frame_id,
                                observation2.image_id,
                                observation2.point2D_idx);
              });

    std::vector<RigPoint> rig_points;
    for (size_t group_begin = 0; group_begin < observations.size();) {
      size_t group_end = group_begin + 1;
      while (group_end < observations.size() &&
             observations[group_end].frame_id ==
                 observations[group_begin].frame_id) {
        ++group_end;
      }
      if (group_end - group_begin < 2) {
        group_begin = group_end;
        continue;
      }

      bool has_distinct_images = false;
      double smallest_cosine = 1.0;
      for (size_t index1 = group_begin; index1 < group_end; ++index1) {
        for (size_t index2 = index1 + 1; index2 < group_end; ++index2) {
          has_distinct_images |=
              observations[index1].image_id != observations[index2].image_id;
          smallest_cosine = std::min(
              smallest_cosine,
              observations[index1].bearing.dot(observations[index2].bearing));
        }
      }
      if (!has_distinct_images || smallest_cosine > minimum_cosine) {
        group_begin = group_end;
        continue;
      }

      Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
      Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
      for (size_t index = group_begin; index < group_end; ++index) {
        const Eigen::Vector3d& bearing = observations[index].bearing;
        const Eigen::Matrix3d ray_projection =
            Eigen::Matrix3d::Identity() - bearing * bearing.transpose();
        normal_matrix += ray_projection;
        right_hand_side += ray_projection * observations[index].center;
      }
      const Eigen::Vector3d xyz = normal_matrix.ldlt().solve(right_hand_side);
      bool has_positive_depth = true;
      for (size_t index = group_begin; index < group_end; ++index) {
        if (observations[index].bearing.dot(xyz - observations[index].center) <=
            0.0) {
          has_positive_depth = false;
          break;
        }
      }
      if (has_positive_depth) {
        rig_points.push_back(RigPoint{observations[group_begin].frame_id, xyz});
      }
      group_begin = group_end;
    }

    for (size_t index = 1; index < rig_points.size(); ++index) {
      const RigPoint& point1 = rig_points[index - 1];
      const RigPoint& point2 = rig_points[index];
      const Eigen::Vector3d center2_from_center1 =
          reconstruction.Frame(point1.frame_id)
                  .RigFromWorld()
                  .rotation()
                  .inverse() *
              point1.xyz -
          reconstruction.Frame(point2.frame_id)
                  .RigFromWorld()
                  .rotation()
                  .inverse() *
              point2.xyz;
      constraints.push_back(FixedRigFrameConstraint{
          point1.frame_id, point2.frame_id, center2_from_center1, 1.0});
    }
  }

  std::sort(constraints.begin(),
            constraints.end(),
            [](const FixedRigFrameConstraint& constraint1,
               const FixedRigFrameConstraint& constraint2) {
              return std::make_tuple(constraint1.frame_id1,
                                     constraint1.frame_id2,
                                     constraint1.center2_from_center1.x(),
                                     constraint1.center2_from_center1.y(),
                                     constraint1.center2_from_center1.z()) <
                     std::make_tuple(constraint2.frame_id1,
                                     constraint2.frame_id2,
                                     constraint2.center2_from_center1.x(),
                                     constraint2.center2_from_center1.y(),
                                     constraint2.center2_from_center1.z());
            });
  std::vector<FixedRigFrameConstraint> aggregated_constraints;
  for (size_t group_begin = 0; group_begin < constraints.size();) {
    size_t group_end = group_begin + 1;
    while (group_end < constraints.size() &&
           constraints[group_end].frame_id1 ==
               constraints[group_begin].frame_id1 &&
           constraints[group_end].frame_id2 ==
               constraints[group_begin].frame_id2) {
      ++group_end;
    }

    Eigen::Vector3d median;
    std::vector<double> coordinates(group_end - group_begin);
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      for (size_t index = group_begin; index < group_end; ++index) {
        coordinates[index - group_begin] =
            constraints[index].center2_from_center1(coordinate);
      }
      median(coordinate) = Median(coordinates);
    }
    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(group_end - group_begin);
    for (size_t index = group_begin; index < group_end; ++index) {
      distances.emplace_back(
          (constraints[index].center2_from_center1 - median).squaredNorm(),
          index);
    }
    std::sort(distances.begin(), distances.end());
    const size_t num_inliers = (9 * distances.size() + 9) / 10;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (size_t index = 0; index < num_inliers; ++index) {
      mean += constraints[distances[index].second].center2_from_center1;
    }
    mean /= num_inliers;
    aggregated_constraints.push_back(
        FixedRigFrameConstraint{constraints[group_begin].frame_id1,
                                constraints[group_begin].frame_id2,
                                mean,
                                std::sqrt(static_cast<double>(num_inliers))});
    group_begin = group_end;
  }
  constraints = std::move(aggregated_constraints);
  for (const FixedRigFrameConstraint& constraint : constraints) {
    adjacency[constraint.frame_id1].push_back(constraint.frame_id2);
    adjacency[constraint.frame_id2].push_back(constraint.frame_id1);
  }

  if (adjacency.size() != frame_centers_.size()) {
    return false;
  }
  std::unordered_set<frame_t> visited;
  std::queue<frame_t> queue;
  queue.push(adjacency.begin()->first);
  while (!queue.empty()) {
    const frame_t frame_id = queue.front();
    queue.pop();
    if (!visited.insert(frame_id).second) {
      continue;
    }
    for (const frame_t neighbor : adjacency.at(frame_id)) {
      queue.push(neighbor);
    }
  }
  return visited.size() == frame_centers_.size();
}

bool GlobalPositioner::SolveFixedRigFramePositions(
    Reconstruction& reconstruction,
    const std::vector<FixedRigFrameConstraint>& constraints) {
  for (const FixedRigFrameConstraint& constraint : constraints) {
    problem_->AddResidualBlock(
        FixedRigFramePositionCostFunctor::Create(
            constraint.center2_from_center1, constraint.weight),
        loss_function_.get(),
        frame_centers_.at(constraint.frame_id1).data(),
        frame_centers_.at(constraint.frame_id2).data());
  }
  const auto anchor =
      std::min_element(frame_centers_.begin(),
                       frame_centers_.end(),
                       [](const auto& frame1, const auto& frame2) {
                         return frame1.first < frame2.first;
                       });
  problem_->SetParameterBlockConstant(anchor->second.data());

  ceres::Solver::Options solver_options = options_.solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.linear_solver_ordering.reset();
  solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  Timer stage_timer;
  stage_timer.Start();
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);
  LOG(INFO) << "Fixed-rig frame positioning done in "
            << stage_timer.ElapsedSeconds() << " seconds";
  LOG(INFO) << (VLOG_IS_ON(2) ? summary.FullReport() : summary.BriefReport());

  stage_timer.Restart();
  ConvertBackResults(reconstruction);
  RefineFixedRigPoints(reconstruction);
  LOG(INFO) << "Fixed-rig point refinement done in "
            << stage_timer.ElapsedSeconds() << " seconds";
  return summary.IsSolutionUsable();
}

void GlobalPositioner::ParameterizeVariables(Reconstruction& reconstruction) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // Initialize cams_in_rig_ with random values if optimizing positions.
  if (options_.optimize_positions) {
    for (auto& [sensor_id, center] : cams_in_rig_) {
      if (problem_->HasParameterBlock(center.data())) {
        center = RandVector3d(-1, 1);
      }
    }
  }

  // If not optimizing positions, set frame centers to be constant.
  if (!options_.optimize_positions) {
    for (auto& [frame_id, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
      }
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
      }
    }
  }
  // Set the first scale to be constant to remove the gauge ambiguity.
  for (double& scale : scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      problem_->SetParameterBlockConstant(&scale);
      break;
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  // TODO: Investigate whether the direct solver should be chosen
  // adaptively based on problem scale.
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  // Convert optimized frame centers back to rig_from_world translations.
  for (const auto& [frame_id, center] : frame_centers_) {
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
  }

  for (const auto& [sensor_id, center] : cams_in_rig_) {
    // Find the rig containing this sensor.
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      if (!rig.HasSensor(sensor_id)) {
        continue;
      }
      Rigid3d& sensor_from_rig =
          reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
      sensor_from_rig.translation() = sensor_from_rig.rotation() * -center;
      break;
    }
  }
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors,
                          double min_tri_angle_deg) {
  GlobalPositioner positioner(options, min_tri_angle_deg);
  return positioner.Solve(pose_graph, reconstruction, pose_priors);
}

}  // namespace colmap
