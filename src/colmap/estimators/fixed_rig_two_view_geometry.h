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

#pragma once

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/types.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/types.h"

#include <utility>
#include <vector>

namespace colmap {

// Match-aligned observations for one image pair between two fixed rigs.
struct FixedRigMatchedPair {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Camera camera1;
  Camera camera2;
  FeatureMatches matches;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
};

// Estimate one metric relative pose shared by all matched camera pairs. PoseLib
// samples a balanced, bounded subset; the returned per-image geometries are
// scored over every input correspondence in pixel-space tangent Sampson error.
std::vector<std::pair<std::pair<image_t, image_t>, TwoViewGeometry>>
EstimateFixedRigTwoViewGeometries(const Rig& rig1,
                                  const Rig& rig2,
                                  const std::vector<FixedRigMatchedPair>& pairs,
                                  const TwoViewGeometryOptions& options,
                                  size_t max_num_ransac_matches);

}  // namespace colmap
