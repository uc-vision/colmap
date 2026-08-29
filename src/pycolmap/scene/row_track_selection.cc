#include "pycolmap/pybind11_extension.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace pybind11::literals;
namespace py = pybind11;

namespace {

constexpr uint32_t kSpatialGridSide = 4;
constexpr uint32_t kSpatialCellCount = kSpatialGridSide * kSpatialGridSide;

using FloatArray = py::array_t<float, py::array::c_style>;
using Int32Array = py::array_t<int32_t, py::array::c_style>;
using Int64Array = py::array_t<int64_t, py::array::c_style>;
using UInt16Array = py::array_t<uint16_t, py::array::c_style>;
using UInt32Array = py::array_t<uint32_t, py::array::c_style>;
using FloatMatrix = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;

struct RowTrackSelectionSource {
  RowTrackSelectionSource(Int64Array observation_offsets,
                          Int32Array observation_image_indices,
                          Int32Array observation_camera_indices,
                          FloatArray observation_xy,
                          UInt32Array row_point_indices,
                          Int32Array image_to_global_index,
                          FloatArray camera_dimensions)
      : observation_offsets(std::move(observation_offsets)),
        observation_image_indices(std::move(observation_image_indices)),
        observation_camera_indices(std::move(observation_camera_indices)),
        observation_xy(std::move(observation_xy)),
        row_point_indices(std::move(row_point_indices)),
        image_to_global_index(std::move(image_to_global_index)),
        camera_dimensions(std::move(camera_dimensions)) {}

  Int64Array observation_offsets;
  Int32Array observation_image_indices;
  Int32Array observation_camera_indices;
  FloatArray observation_xy;
  UInt32Array row_point_indices;
  Int32Array image_to_global_index;
  FloatArray camera_dimensions;
};

py::array_t<uint32_t> SelectRowBundlePoints(
    const std::vector<const RowTrackSelectionSource*>& sources,
    const UInt16Array& source_support,
    int64_t minimum_track_length,
    uint32_t tracks_per_spatial_cell) {
  std::vector<uint32_t> selected_indices;
  {
    py::gil_scoped_release release;
    const size_t point_count = source_support.shape(0);
    std::vector<uint8_t> selected(point_count, uint8_t{0});
    const uint16_t* support = source_support.data();
    for (uint32_t point = 0; point < point_count; ++point) {
      selected[point] = support[point] > 1;
    }

    int32_t global_image_count = 0;
    for (const RowTrackSelectionSource* source : sources) {
      const int32_t* image_to_global_index =
          source->image_to_global_index.data();
      for (py::ssize_t image = 0;
           image < source->image_to_global_index.shape(0);
           ++image) {
        global_image_count =
            std::max(global_image_count, image_to_global_index[image] + 1);
      }
    }
    std::vector<uint32_t> stratum_count(global_image_count * kSpatialCellCount);

    for (const RowTrackSelectionSource* source : sources) {
      std::fill(stratum_count.begin(), stratum_count.end(), uint32_t{0});
      const int64_t* observation_offsets = source->observation_offsets.data();
      const int32_t* observation_image_indices =
          source->observation_image_indices.data();
      const int32_t* observation_camera_indices =
          source->observation_camera_indices.data();
      const uint32_t* row_point_indices = source->row_point_indices.data();
      const int32_t* image_to_global_index =
          source->image_to_global_index.data();
      const Eigen::Map<const FloatMatrix> observation_xy(
          source->observation_xy.data(), source->observation_xy.shape(0), 2);
      const Eigen::Map<const FloatMatrix> camera_dimensions(
          source->camera_dimensions.data(),
          source->camera_dimensions.shape(0),
          2);
      const uint32_t track_count = source->observation_offsets.shape(0) - 1;
      for (uint32_t track = 0; track < track_count; ++track) {
        const uint32_t row_point = row_point_indices[track];
        if (support[row_point] > 1 ||
            observation_offsets[track + 1] - observation_offsets[track] <
                minimum_track_length) {
          continue;
        }
        for (int64_t observation = observation_offsets[track];
             observation < observation_offsets[track + 1];
             ++observation) {
          const int32_t global_image =
              image_to_global_index[observation_image_indices[observation]];
          const int32_t camera = observation_camera_indices[observation];
          const Eigen::RowVector2i cell =
              (observation_xy.row(observation).array() *
               static_cast<float>(kSpatialGridSide) /
               camera_dimensions.row(camera).array())
                  .cast<int32_t>()
                  .matrix();
          const uint32_t stratum = global_image * kSpatialCellCount +
                                   cell.y() * kSpatialGridSide + cell.x();
          if (stratum_count[stratum] < tracks_per_spatial_cell) {
            ++stratum_count[stratum];
            selected[row_point] = 1;
          }
        }
      }
    }

    selected_indices.reserve(
        std::count(selected.begin(), selected.end(), static_cast<uint8_t>(1)));
    for (uint32_t point = 0; point < point_count; ++point) {
      if (selected[point] != 0) {
        selected_indices.push_back(point);
      }
    }
  }

  py::array_t<uint32_t> output(selected_indices.size());
  std::copy(
      selected_indices.begin(), selected_indices.end(), output.mutable_data());
  return output;
}

}  // namespace

void BindRowTrackSelection(py::module& m) {
  py::classh<RowTrackSelectionSource>(m, "RowTrackSelectionSource")
      .def(py::init<Int64Array,
                    Int32Array,
                    Int32Array,
                    FloatArray,
                    UInt32Array,
                    Int32Array,
                    FloatArray>(),
           "observation_offsets"_a,
           "observation_image_indices"_a,
           "observation_camera_indices"_a,
           "observation_xy"_a,
           "row_point_indices"_a,
           "image_to_global_index"_a,
           "camera_dimensions"_a)
      .def_readonly("observation_offsets",
                    &RowTrackSelectionSource::observation_offsets)
      .def_readonly("observation_image_indices",
                    &RowTrackSelectionSource::observation_image_indices)
      .def_readonly("observation_camera_indices",
                    &RowTrackSelectionSource::observation_camera_indices)
      .def_readonly("observation_xy", &RowTrackSelectionSource::observation_xy)
      .def_readonly("row_point_indices",
                    &RowTrackSelectionSource::row_point_indices)
      .def_readonly("image_to_global_index",
                    &RowTrackSelectionSource::image_to_global_index)
      .def_readonly("camera_dimensions",
                    &RowTrackSelectionSource::camera_dimensions);

  m.def("select_row_bundle_points",
        &SelectRowBundlePoints,
        "sources"_a,
        "source_support"_a,
        "minimum_track_length"_a,
        "tracks_per_spatial_cell"_a,
        "Select bounded spatial row points in stable track order.");
}
