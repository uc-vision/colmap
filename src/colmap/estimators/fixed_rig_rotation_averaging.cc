#include "colmap/estimators/fixed_rig_rotation_averaging.h"

#include "colmap/math/math.h"
#include "colmap/util/logging.h"

#include <limits>

namespace colmap {
namespace {

Rigid3d CamFromRig(const Image& image, const Reconstruction& reconstruction) {
  if (image.IsRefInFrame()) {
    return Rigid3d();
  }
  return reconstruction.Rig(image.FramePtr()->RigId())
      .SensorFromRig(image.CameraPtr()->SensorId());
}

struct FixedRigRotationSample {
  image_pair_t image_pair_id;
  Eigen::Quaterniond frame2_from_frame1;
};

}  // namespace

void FilterFixedRigRotationOutliers(PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction,
                                    const double max_rotation_error_deg) {
  NodeHashMap<image_pair_t, std::vector<FixedRigRotationSample>>
      frame_pair_samples;
  for (const auto& [image_pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(image_pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    const frame_t frame_id1 = image1.FrameId();
    const frame_t frame_id2 = image2.FrameId();
    if (frame_id1 == frame_id2) {
      continue;
    }

    Eigen::Quaterniond frame2_from_frame1 =
        CamFromRig(image2, reconstruction).rotation().inverse() *
        edge.cam2_from_cam1.rotation() *
        CamFromRig(image1, reconstruction).rotation();
    if (frame_id2 < frame_id1) {
      frame2_from_frame1 = frame2_from_frame1.inverse();
    }
    frame_pair_samples[ImagePairToPairId(frame_id1, frame_id2)].push_back(
        {image_pair_id, frame2_from_frame1});
  }

  const double max_rotation_error =
      max_rotation_error_deg > 0 ? DegToRad(max_rotation_error_deg)
                                 : std::numeric_limits<double>::infinity();
  size_t num_image_pairs = 0;
  size_t num_rejected = 0;
  for (const auto& frame_pair : frame_pair_samples) {
    const auto& samples = frame_pair.second;
    size_t medoid_index = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t sample_index = 0; sample_index < samples.size();
         ++sample_index) {
      double score = 0;
      for (const auto& sample : samples) {
        score += samples[sample_index].frame2_from_frame1.angularDistance(
            sample.frame2_from_frame1);
      }
      if (score < best_score) {
        best_score = score;
        medoid_index = sample_index;
      }
    }

    const Eigen::Quaterniond& medoid = samples[medoid_index].frame2_from_frame1;
    for (const auto& sample : samples) {
      ++num_image_pairs;
      if (medoid.angularDistance(sample.frame2_from_frame1) >
          max_rotation_error) {
        pose_graph.SetInvalidEdge(sample.image_pair_id);
        ++num_rejected;
      }
    }
  }

  LOG(INFO) << "Filtered " << frame_pair_samples.size()
            << " fixed-rig frame-pair rotation groups containing "
            << num_image_pairs << " image pairs (rejected " << num_rejected
            << ")";
}

}  // namespace colmap
