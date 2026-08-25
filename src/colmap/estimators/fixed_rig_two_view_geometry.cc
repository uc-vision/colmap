// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/fixed_rig_two_view_geometry.h"

#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <limits>

#include <PoseLib/robust.h>

namespace colmap {
namespace {

struct PoseLibRig {
  FlatHashMap<camera_t, size_t> camera_indices;
  std::vector<poselib::CameraPose> cams_from_rig;
  std::vector<poselib::Camera> cameras;
};

Rigid3d CameraFromRig(const Rig& rig, const Camera& camera) {
  return rig.IsRefSensor(camera.SensorId())
             ? Rigid3d()
             : rig.SensorFromRig(camera.SensorId());
}

size_t AddPoseLibCamera(const Rig& rig,
                        const Camera& camera,
                        PoseLibRig* poselib_rig) {
  const auto [it, inserted] = poselib_rig->camera_indices.emplace(
      camera.camera_id, poselib_rig->cameras.size());
  if (inserted) {
    poselib_rig->cams_from_rig.push_back(
        ConvertRigid3dToPoseLibPose(CameraFromRig(rig, camera)));
    poselib_rig->cameras.push_back(ConvertCameraToPoseLibCamera(camera));
  }
  return it->second;
}

std::vector<poselib::PairwiseMatches> BalanceRigMatches(
    const std::vector<poselib::PairwiseMatches>& matches,
    const size_t max_num_matches) {
  std::vector<const poselib::PairwiseMatches*> nonempty_matches;
  size_t num_matches = 0;
  for (const auto& pair_matches : matches) {
    if (pair_matches.x1.empty()) {
      continue;
    }
    nonempty_matches.push_back(&pair_matches);
    num_matches += pair_matches.x1.size();
  }

  if (nonempty_matches.size() < 2) {
    return {};
  }
  const auto anchor_it =
      std::max_element(nonempty_matches.begin(),
                       nonempty_matches.end(),
                       [](const auto* lhs, const auto* rhs) {
                         return lhs->x1.size() < rhs->x1.size();
                       });
  const size_t anchor_idx = anchor_it - nonempty_matches.begin();
  if ((*anchor_it)->x1.size() < 5) {
    return {};
  }
  if (num_matches <= max_num_matches) {
    std::vector<poselib::PairwiseMatches> result;
    result.reserve(nonempty_matches.size());
    for (const auto* pair_matches : nonempty_matches) {
      result.push_back(*pair_matches);
    }
    return result;
  }

  std::vector<size_t> quotas(nonempty_matches.size(), 0);
  quotas[anchor_idx] = 5;
  quotas[anchor_idx == 0 ? 1 : 0] = 1;
  size_t remaining = max_num_matches - 6;
  size_t camera_pair_idx = 0;
  while (remaining > 0) {
    if (quotas[camera_pair_idx] <
        nonempty_matches[camera_pair_idx]->x1.size()) {
      ++quotas[camera_pair_idx];
      --remaining;
    }
    camera_pair_idx = (camera_pair_idx + 1) % nonempty_matches.size();
  }

  std::vector<poselib::PairwiseMatches> result;
  result.reserve(nonempty_matches.size());
  for (size_t pair_idx = 0; pair_idx < nonempty_matches.size(); ++pair_idx) {
    const auto& source = *nonempty_matches[pair_idx];
    poselib::PairwiseMatches selected;
    selected.cam_id1 = source.cam_id1;
    selected.cam_id2 = source.cam_id2;
    selected.x1.reserve(quotas[pair_idx]);
    selected.x2.reserve(quotas[pair_idx]);
    for (size_t i = 0; i < quotas[pair_idx]; ++i) {
      const size_t source_idx = i * source.x1.size() / quotas[pair_idx];
      selected.x1.push_back(source.x1[source_idx]);
      selected.x2.push_back(source.x2[source_idx]);
    }
    if (!selected.x1.empty()) {
      result.push_back(std::move(selected));
    }
  }
  return result;
}

poselib::RelativePoseOptions PoseLibRelativePoseOptions(
    const RANSACOptions& options) {
  poselib::RelativePoseOptions poselib_options;
  poselib_options.max_error = options.max_error;
  poselib_options.ransac.min_iterations = options.min_num_trials;
  poselib_options.ransac.max_iterations = options.max_num_trials;
  poselib_options.ransac.success_prob = options.confidence;
  poselib_options.ransac.dyn_num_trials_mult =
      options.dyn_num_trials_multiplier;
  poselib_options.ransac.seed =
      options.random_seed >= 0
          ? static_cast<unsigned long>(options.random_seed)
          : RandomUniformInteger<unsigned long>(
                0, std::numeric_limits<unsigned long>::max());
  return poselib_options;
}

}  // namespace

std::vector<std::pair<std::pair<image_t, image_t>, TwoViewGeometry>>
EstimateFixedRigTwoViewGeometries(const Rig& rig1,
                                  const Rig& rig2,
                                  const std::vector<FixedRigMatchedPair>& pairs,
                                  const TwoViewGeometryOptions& options,
                                  const size_t max_num_ransac_matches) {
  THROW_CHECK(options.Check());
  THROW_CHECK_GE(max_num_ransac_matches, 6);

  PoseLibRig poselib_rig1;
  PoseLibRig poselib_rig2;
  std::vector<poselib::PairwiseMatches> pairwise_matches;
  pairwise_matches.reserve(pairs.size());

  FlatHashSet<image_pair_t> image_pairs;
  image_pairs.reserve(pairs.size());
  for (const FixedRigMatchedPair& pair : pairs) {
    THROW_CHECK(
        image_pairs.insert(ImagePairToPairId(pair.image_id1, pair.image_id2))
            .second)
        << "Duplicate image pair";
    THROW_CHECK_EQ(pair.matches.size(), pair.points1.size());
    THROW_CHECK_EQ(pair.matches.size(), pair.points2.size());

    poselib::PairwiseMatches pair_matches;
    pair_matches.cam_id1 = AddPoseLibCamera(rig1, pair.camera1, &poselib_rig1);
    pair_matches.cam_id2 = AddPoseLibCamera(rig2, pair.camera2, &poselib_rig2);
    pair_matches.x1 = pair.points1;
    pair_matches.x2 = pair.points2;
    pairwise_matches.push_back(std::move(pair_matches));
  }

  const std::vector<poselib::PairwiseMatches> ransac_matches =
      BalanceRigMatches(pairwise_matches, max_num_ransac_matches);
  if (ransac_matches.empty()) {
    return {};
  }

  poselib::CameraPose poselib_rig2_from_rig1;
  std::vector<std::vector<char>> ransac_inliers;
  const poselib::RansacStats ransac_stats =
      poselib::estimate_generalized_relative_pose(
          ransac_matches,
          poselib_rig1.cams_from_rig,
          poselib_rig1.cameras,
          poselib_rig2.cams_from_rig,
          poselib_rig2.cameras,
          PoseLibRelativePoseOptions(options.ransac_options),
          &poselib_rig2_from_rig1,
          &ransac_inliers);
  if (ransac_stats.num_inliers < 6) {
    return {};
  }
  const Rigid3d rig2_from_rig1 =
      ConvertPoseLibPoseToRigid3d(poselib_rig2_from_rig1);

  size_t num_inliers = 0;
  std::vector<std::pair<std::pair<image_t, image_t>, TwoViewGeometry>>
      two_view_geometries;
  two_view_geometries.reserve(pairs.size());
  const double max_squared_error =
      options.ransac_options.max_error * options.ransac_options.max_error;
  for (const FixedRigMatchedPair& pair : pairs) {
    const Rigid3d cam2_from_cam1 = CameraFromRig(rig2, pair.camera2) *
                                   rig2_from_rig1 *
                                   Inverse(CameraFromRig(rig1, pair.camera1));
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

    TwoViewGeometry geometry;
    geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED_RIG;
    for (size_t i = 0; i < pair.matches.size(); ++i) {
      const CamRayWithJac ray1 =
          pair.camera1.CamRayFromImgWithJac(pair.points1[i])
              .value_or(CamRayWithJac::Zero());
      const CamRayWithJac ray2 =
          pair.camera2.CamRayFromImgWithJac(pair.points2[i])
              .value_or(CamRayWithJac::Zero());
      if (ComputeSquaredTangentSampsonError(ray1, ray2, E) <=
          max_squared_error) {
        geometry.inlier_matches.push_back(pair.matches[i]);
      }
    }
    num_inliers += geometry.inlier_matches.size();
    const std::pair<image_t, image_t> image_pair(pair.image_id1,
                                                 pair.image_id2);
    if (geometry.inlier_matches.empty()) {
      two_view_geometries.emplace_back(image_pair, TwoViewGeometry());
    } else {
      geometry.cam2_from_cam1 = cam2_from_cam1;
      geometry.E = E;
      two_view_geometries.emplace_back(image_pair, std::move(geometry));
    }
  }
  if (num_inliers < static_cast<size_t>(options.min_num_inliers)) {
    return {};
  }
  return two_view_geometries;
}

}  // namespace colmap
