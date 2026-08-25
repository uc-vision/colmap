#pragma once

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

// Reject per-image relative rotations that disagree with the robust consensus
// for their calibrated-rig frame pair.
void FilterFixedRigRotationOutliers(PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction,
                                    double max_rotation_error_deg);

}  // namespace colmap
