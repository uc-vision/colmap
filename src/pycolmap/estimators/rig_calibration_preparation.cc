#include "pycolmap/estimators/rig_calibration_preparation.h"

#include "colmap/estimators/triangulation.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;
using Uint64Array = py::array_t<uint64_t, py::array::c_style>;

struct PackedRigCalibrationPreparation {
  Uint64Array track_observation_offsets;
  DoubleArray xyz;
  Uint32Array frame_indices;
  Uint32Array camera_ids;
  DoubleArray xy;
  size_t attempted_tracks = 0;
  size_t retained_tracks = 0;
};

struct NativeRigCalibrationPreparation {
  std::vector<uint64_t> track_observation_offsets = {0};
  std::vector<double> xyz;
  std::vector<uint32_t> frame_indices;
  std::vector<uint32_t> camera_ids;
  std::vector<double> xy;
  size_t attempted_tracks = 0;
  size_t retained_tracks = 0;
};

struct PackedPreparationInput {
  const uint32_t* component_observation_offsets;
  const uint64_t* component_observation_codes;
  const uint32_t* ordered_component_indices;
  size_t num_ordered_components;
  size_t max_tracks;
  const uint32_t* image_ids;
  const uint32_t* image_frame_indices;
  const uint32_t* image_camera_ids;
  const uint32_t* image_keypoint_offsets;
  size_t num_images;
  const double* keypoints;
  size_t num_keypoints;
  const double* rigs_from_group;
  size_t group_size;
};

struct PreparedImage {
  uint32_t frame_idx;
  camera_t camera_id;
  uint32_t keypoint_offset;
  uint32_t num_keypoints;
  const Camera* camera;
  Rigid3d cam_from_group;
  Eigen::Matrix3x4d cam_from_group_matrix;
};

template <typename T>
py::array_t<T> ArrayFromVector(std::vector<T>&& values) {
  py::array_t<T> array(values.size());
  if (!values.empty()) {
    std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
  }
  return array;
}

DoubleArray MatrixArrayFromVector(std::vector<double>&& values,
                                  const size_t num_rows,
                                  const size_t num_columns) {
  DoubleArray array(std::vector<ssize_t>{static_cast<ssize_t>(num_rows),
                                         static_cast<ssize_t>(num_columns)});
  if (!values.empty()) {
    std::memcpy(
        array.mutable_data(), values.data(), values.size() * sizeof(double));
  }
  return array;
}

std::vector<Rigid3d> ReadRigsFromGroup(const double* matrices,
                                       const size_t group_size) {
  std::vector<Rigid3d> rigs_from_group;
  rigs_from_group.reserve(group_size);
  for (size_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
    Eigen::Matrix3x4d matrix;
    for (size_t row = 0; row < 3; ++row) {
      for (size_t column = 0; column < 4; ++column) {
        matrix(row, column) = matrices[frame_idx * 12 + row * 4 + column];
      }
    }
    rigs_from_group.push_back(Rigid3d::FromMatrix(matrix));
  }
  return rigs_from_group;
}

NativeRigCalibrationPreparation PrepareRigCalibrationGroup(
    const Reconstruction& reconstruction,
    const rig_t rig_id,
    const PackedPreparationInput& input) {
  const Rig& rig = reconstruction.Rig(rig_id);
  const std::vector<Rigid3d> rigs_from_group =
      ReadRigsFromGroup(input.rigs_from_group, input.group_size);
  std::vector<PreparedImage> images;
  images.reserve(input.num_images);
  FlatHashMap<image_t, size_t> image_slots;
  image_slots.reserve(input.num_images);
  std::vector<Eigen::Vector3d> keypoint_rays(input.num_keypoints);
  std::vector<uint8_t> keypoint_ray_valid(input.num_keypoints, false);
  for (size_t image_idx = 0; image_idx < input.num_images; ++image_idx) {
    const image_t image_id = input.image_ids[image_idx];
    const uint32_t frame_idx = input.image_frame_indices[image_idx];
    THROW_CHECK_LT(frame_idx, rigs_from_group.size());
    const camera_t camera_id = input.image_camera_ids[image_idx];
    const Camera& camera = reconstruction.Camera(camera_id);
    const sensor_t sensor_id(SensorType::CAMERA, camera_id);
    const Rigid3d sensor_from_rig =
        rig.IsRefSensor(sensor_id) ? Rigid3d() : rig.SensorFromRig(sensor_id);
    const Rigid3d cam_from_group = sensor_from_rig * rigs_from_group[frame_idx];
    const uint32_t keypoint_begin = input.image_keypoint_offsets[image_idx];
    const uint32_t keypoint_end = input.image_keypoint_offsets[image_idx + 1];
    images.push_back({frame_idx,
                      camera_id,
                      keypoint_begin,
                      keypoint_end - keypoint_begin,
                      &camera,
                      cam_from_group,
                      cam_from_group.ToMatrix()});
    image_slots.emplace(image_id, image_idx);
    for (uint32_t keypoint_idx = keypoint_begin; keypoint_idx < keypoint_end;
         ++keypoint_idx) {
      const Eigen::Vector2d point(input.keypoints[2 * keypoint_idx],
                                  input.keypoints[2 * keypoint_idx + 1]);
      const std::optional<Eigen::Vector3d> cam_ray =
          camera.CamRayFromImg(point);
      if (cam_ray.has_value()) {
        keypoint_rays[keypoint_idx] = *cam_ray;
        keypoint_ray_valid[keypoint_idx] = true;
      }
    }
  }

  NativeRigCalibrationPreparation result;
  result.xyz.reserve(3 *
                     std::min(input.max_tracks, input.num_ordered_components));
  result.track_observation_offsets.reserve(
      std::min(input.max_tracks, input.num_ordered_components) + 1);

  std::vector<Eigen::Matrix3x4d> cam_matrices;
  std::vector<Eigen::Vector3d> cam_rays;
  std::vector<Eigen::Vector2d> points;
  std::vector<Rigid3d> cams_from_group;
  std::vector<const Camera*> cameras;
  std::vector<uint32_t> frame_indices;
  std::vector<uint32_t> camera_ids;
  std::vector<char> inlier_mask;
  EstimateTriangulationOptions triangulation_options;
  triangulation_options.ransac_options.random_seed = 42;

  for (size_t order_idx = 0; order_idx < input.num_ordered_components &&
                             result.retained_tracks < input.max_tracks;
       ++order_idx) {
    ++result.attempted_tracks;
    const uint32_t component_idx = input.ordered_component_indices[order_idx];
    const uint32_t observation_begin =
        input.component_observation_offsets[component_idx];
    const uint32_t observation_end =
        input.component_observation_offsets[component_idx + 1];
    const size_t num_observations = observation_end - observation_begin;
    cam_matrices.clear();
    cam_rays.clear();
    points.clear();
    cams_from_group.clear();
    cameras.clear();
    frame_indices.clear();
    camera_ids.clear();
    cam_matrices.reserve(num_observations);
    cam_rays.reserve(num_observations);
    points.reserve(num_observations);
    cams_from_group.reserve(num_observations);
    cameras.reserve(num_observations);
    frame_indices.reserve(num_observations);
    camera_ids.reserve(num_observations);

    bool observations_valid = true;
    for (uint32_t observation_idx = observation_begin;
         observation_idx < observation_end;
         ++observation_idx) {
      const uint64_t code = input.component_observation_codes[observation_idx];
      const image_t image_id = static_cast<image_t>(code >> 32);
      const uint32_t point_idx = static_cast<uint32_t>(code);
      const size_t image_idx = image_slots.at(image_id);
      const PreparedImage& image = images[image_idx];
      THROW_CHECK_LT(point_idx, image.num_keypoints);
      const size_t keypoint_idx = image.keypoint_offset + point_idx;
      if (!keypoint_ray_valid[keypoint_idx]) {
        observations_valid = false;
        break;
      }
      cam_matrices.push_back(image.cam_from_group_matrix);
      cam_rays.push_back(keypoint_rays[keypoint_idx]);
      points.emplace_back(input.keypoints[2 * keypoint_idx],
                          input.keypoints[2 * keypoint_idx + 1]);
      cams_from_group.push_back(image.cam_from_group);
      cameras.push_back(image.camera);
      frame_indices.push_back(image.frame_idx);
      camera_ids.push_back(image.camera_id);
    }
    if (!observations_valid) {
      continue;
    }

    Eigen::Vector3d xyz;
    bool success = TriangulateMultiViewPoint(
        span<const Eigen::Matrix3x4d>(cam_matrices.data(), cam_matrices.size()),
        span<const Eigen::Vector3d>(cam_rays.data(), cam_rays.size()),
        &xyz);
    if (success) {
      for (size_t observation_idx = 0; observation_idx < num_observations;
           ++observation_idx) {
        if (cameras[observation_idx]->IsPerspective()) {
          if (!HasPointPositiveDepth(cam_matrices[observation_idx], xyz)) {
            success = false;
            break;
          }
        } else if ((cams_from_group[observation_idx] * xyz)
                       .dot(cam_rays[observation_idx]) <= 0.0) {
          success = false;
          break;
        }
      }
    }
    if (!success) {
      success = EstimateTriangulation(triangulation_options,
                                      points,
                                      cams_from_group,
                                      cameras,
                                      &inlier_mask,
                                      &xyz);
    }
    if (!success) {
      continue;
    }

    result.xyz.insert(result.xyz.end(), xyz.data(), xyz.data() + 3);
    result.frame_indices.insert(
        result.frame_indices.end(), frame_indices.begin(), frame_indices.end());
    result.camera_ids.insert(
        result.camera_ids.end(), camera_ids.begin(), camera_ids.end());
    for (const Eigen::Vector2d& point : points) {
      result.xy.push_back(point.x());
      result.xy.push_back(point.y());
    }
    ++result.retained_tracks;
    result.track_observation_offsets.push_back(result.frame_indices.size());
  }
  return result;
}

PackedRigCalibrationPreparation PyPrepareRigCalibrationGroup(
    const Reconstruction& reconstruction,
    const rig_t rig_id,
    const Uint32Array& component_observation_offsets,
    const Uint64Array& component_observation_codes,
    const Uint32Array& ordered_component_indices,
    const size_t max_tracks,
    const Uint32Array& image_ids,
    const Uint32Array& image_frame_indices,
    const Uint32Array& image_camera_ids,
    const Uint32Array& image_keypoint_offsets,
    const DoubleArray& keypoints,
    const DoubleArray& rigs_from_group) {
  THROW_CHECK_EQ(component_observation_offsets.ndim(), 1);
  THROW_CHECK_GE(component_observation_offsets.shape(0), 1);
  const size_t num_components = component_observation_offsets.shape(0) - 1;
  THROW_CHECK_EQ(component_observation_codes.ndim(), 1);
  THROW_CHECK_EQ(ordered_component_indices.ndim(), 1);
  THROW_CHECK_EQ(image_ids.ndim(), 1);
  const size_t num_images = image_ids.shape(0);
  THROW_CHECK_EQ(image_frame_indices.ndim(), 1);
  THROW_CHECK_EQ(image_frame_indices.shape(0), num_images);
  THROW_CHECK_EQ(image_camera_ids.ndim(), 1);
  THROW_CHECK_EQ(image_camera_ids.shape(0), num_images);
  THROW_CHECK_EQ(image_keypoint_offsets.ndim(), 1);
  THROW_CHECK_EQ(image_keypoint_offsets.shape(0), num_images + 1);
  THROW_CHECK_EQ(keypoints.ndim(), 2);
  THROW_CHECK_EQ(keypoints.shape(1), 2);
  const size_t num_keypoints = keypoints.shape(0);
  THROW_CHECK_EQ(rigs_from_group.ndim(), 3);
  THROW_CHECK_GE(rigs_from_group.shape(0), 2);
  THROW_CHECK_EQ(rigs_from_group.shape(1), 3);
  THROW_CHECK_EQ(rigs_from_group.shape(2), 4);
  const size_t group_size = static_cast<size_t>(rigs_from_group.shape(0));

  const auto component_offsets_view =
      component_observation_offsets.unchecked<1>();
  THROW_CHECK_EQ(component_offsets_view(0), 0);
  THROW_CHECK_EQ(component_offsets_view(num_components),
                 component_observation_codes.shape(0));
  for (size_t component_idx = 0; component_idx < num_components;
       ++component_idx) {
    THROW_CHECK_LE(component_offsets_view(component_idx),
                   component_offsets_view(component_idx + 1));
  }
  const auto ordered_indices_view = ordered_component_indices.unchecked<1>();
  for (size_t order_idx = 0;
       order_idx < static_cast<size_t>(ordered_component_indices.shape(0));
       ++order_idx) {
    THROW_CHECK_LT(ordered_indices_view(order_idx), num_components);
  }
  const auto image_ids_view = image_ids.unchecked<1>();
  for (size_t image_idx = 1; image_idx < num_images; ++image_idx) {
    THROW_CHECK_LT(image_ids_view(image_idx - 1), image_ids_view(image_idx));
  }
  const auto keypoint_offsets_view = image_keypoint_offsets.unchecked<1>();
  THROW_CHECK_EQ(keypoint_offsets_view(0), 0);
  THROW_CHECK_EQ(keypoint_offsets_view(num_images), num_keypoints);
  for (size_t image_idx = 0; image_idx < num_images; ++image_idx) {
    THROW_CHECK_LE(keypoint_offsets_view(image_idx),
                   keypoint_offsets_view(image_idx + 1));
  }

  const PackedPreparationInput input = {
      component_observation_offsets.data(),
      component_observation_codes.data(),
      ordered_component_indices.data(),
      static_cast<size_t>(ordered_component_indices.shape(0)),
      max_tracks,
      image_ids.data(),
      image_frame_indices.data(),
      image_camera_ids.data(),
      image_keypoint_offsets.data(),
      num_images,
      keypoints.data(),
      num_keypoints,
      rigs_from_group.data(),
      group_size};
  NativeRigCalibrationPreparation native_result;
  {
    py::gil_scoped_release release;
    native_result = PrepareRigCalibrationGroup(reconstruction, rig_id, input);
  }
  const size_t num_observations = native_result.frame_indices.size();
  return {
      ArrayFromVector(std::move(native_result.track_observation_offsets)),
      MatrixArrayFromVector(
          std::move(native_result.xyz), native_result.retained_tracks, 3),
      ArrayFromVector(std::move(native_result.frame_indices)),
      ArrayFromVector(std::move(native_result.camera_ids)),
      MatrixArrayFromVector(std::move(native_result.xy), num_observations, 2),
      native_result.attempted_tracks,
      native_result.retained_tracks};
}

}  // namespace

void BindRigCalibrationPreparation(py::module& m) {
  py::classh<PackedRigCalibrationPreparation>(m,
                                              "PackedRigCalibrationPreparation")
      .def_readonly("track_observation_offsets",
                    &PackedRigCalibrationPreparation::track_observation_offsets)
      .def_readonly("xyz", &PackedRigCalibrationPreparation::xyz)
      .def_readonly("frame_indices",
                    &PackedRigCalibrationPreparation::frame_indices)
      .def_readonly("camera_ids", &PackedRigCalibrationPreparation::camera_ids)
      .def_readonly("xy", &PackedRigCalibrationPreparation::xy)
      .def_readonly("attempted_tracks",
                    &PackedRigCalibrationPreparation::attempted_tracks)
      .def_readonly("retained_tracks",
                    &PackedRigCalibrationPreparation::retained_tracks);

  m.def("prepare_rig_calibration_group",
        &PyPrepareRigCalibrationGroup,
        "reconstruction"_a,
        "rig_id"_a,
        py::arg("component_observation_offsets").noconvert(),
        py::arg("component_observation_codes").noconvert(),
        py::arg("ordered_component_indices").noconvert(),
        "max_tracks"_a,
        py::arg("image_ids").noconvert(),
        py::arg("image_frame_indices").noconvert(),
        py::arg("image_camera_ids").noconvert(),
        py::arg("image_keypoint_offsets").noconvert(),
        py::arg("keypoints").noconvert(),
        py::arg("rigs_from_group").noconvert());
}
