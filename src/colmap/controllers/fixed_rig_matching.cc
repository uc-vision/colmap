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

#include "colmap/controllers/fixed_rig_matching.h"

#include "colmap/controllers/matcher_cache.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/frame.h"
#include "colmap/sensor/rig.h"

namespace colmap {
namespace {

Rigid3d CameraFromRig(const Rig& rig, const sensor_t sensor_id) {
  return rig.IsRefSensor(sensor_id) ? Rigid3d() : rig.SensorFromRig(sensor_id);
}

}  // namespace

std::vector<std::pair<image_t, image_t>> FixedRigIntraFrameImagePairs(
    const std::shared_ptr<FeatureMatcherCache>& cache) {
  std::vector<std::pair<image_t, image_t>> image_pairs;
  for (const frame_t frame_id : cache->GetFrameIds()) {
    std::vector<image_t> image_ids;
    for (const data_t& data_id : cache->GetFrame(frame_id).ImageIds()) {
      image_ids.push_back(data_id.id);
    }
    for (size_t i = 0; i < image_ids.size(); ++i) {
      for (size_t j = i + 1; j < image_ids.size(); ++j) {
        image_pairs.emplace_back(image_ids[i], image_ids[j]);
      }
    }
  }
  return image_pairs;
}

TwoViewGeometry CreateFixedRigGuidedGeometry(
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const Image& image1,
    const Image& image2) {
  const Frame& frame = cache->GetFrame(image1.FrameId());
  const Rig& rig = cache->GetRig(frame.RigId());
  const Rigid3d cam1_from_rig = CameraFromRig(rig, image1.DataId().sensor_id);
  const Rigid3d cam2_from_rig = CameraFromRig(rig, image2.DataId().sensor_id);

  TwoViewGeometry geometry;
  geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED_RIG;
  geometry.cam2_from_cam1 = cam2_from_rig * Inverse(cam1_from_rig);
  geometry.E = EssentialMatrixFromPose(*geometry.cam2_from_cam1);
  return geometry;
}

}  // namespace colmap
