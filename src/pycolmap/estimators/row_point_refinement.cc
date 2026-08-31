#ifdef CASPAR_ENABLED
#include "colmap/estimators/row_bundle_adjustment_caspar.h"
#include "colmap/estimators/row_point_refinement_caspar.h"
#include "colmap/estimators/row_section_bundle_adjustment_caspar.h"
#endif

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

#ifdef CASPAR_ENABLED
using DoubleArray = py::array_t<double, py::array::c_style>;
using BoolArray = py::array_t<bool, py::array::c_style>;
using FloatArray = py::array_t<float, py::array::c_style>;
using StridedFloatArray = py::array_t<float, 0>;
using Int32Array = py::array_t<int32_t, py::array::c_style>;
using Int64Array = py::array_t<int64_t, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;

struct PyCasparRowPointRefinementResult {
  py::array_t<float> points;
  py::array_t<float> colors;
  size_t chunk_count;
  int iteration_count;
  int maximum_iterations_run;
  double runtime_seconds;
  double initial_score;
  double final_score;
  double preparation_seconds;
  double packing_seconds;
  double optimization_seconds;
  double validation_seconds;
  CasparReprojectionErrorSummary reprojection;
};

struct PyCasparRowTrackSource {
  PyCasparRowTrackSource(StridedFloatArray points,
                         Int64Array source_point_indices,
                         Int64Array observation_offsets,
                         Int32Array observation_image_indices,
                         FloatArray observation_xy,
                         Uint32Array duplicate_observation_indices,
                         Uint32Array image_rows)
      : points(std::move(points)),
        source_point_indices(std::move(source_point_indices)),
        observation_offsets(std::move(observation_offsets)),
        observation_image_indices(std::move(observation_image_indices)),
        observation_xy(std::move(observation_xy)),
        duplicate_observation_indices(std::move(duplicate_observation_indices)),
        image_rows(std::move(image_rows)) {}

  StridedFloatArray points;
  Int64Array source_point_indices;
  Int64Array observation_offsets;
  Int32Array observation_image_indices;
  FloatArray observation_xy;
  Uint32Array duplicate_observation_indices;
  Uint32Array image_rows;
};

struct PyCasparRowPointRefinementSource {
  PyCasparRowPointRefinementSource(const PyCasparRowTrackSource* tracks,
                                   StridedFloatArray colors,
                                   DoubleArray solved_world_from_source_world)
      : tracks(tracks),
        colors(std::move(colors)),
        solved_world_from_source_world(
            std::move(solved_world_from_source_world)) {}

  const PyCasparRowTrackSource* tracks;
  StridedFloatArray colors;
  DoubleArray solved_world_from_source_world;
};

struct PyCasparRowSectionResult {
  std::vector<Rigid3d> rigs_from_world;
  py::array_t<uint32_t> row_point_indices;
  py::array_t<float> points;
  size_t observation_count;
  size_t active_observation_count;
  double preparation_seconds;
  double optimization_seconds;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

struct PyCasparRowBundleResult {
  std::vector<Rigid3d> rigs_from_world;
  py::array_t<uint32_t> row_point_indices;
  py::array_t<float> points;
  double sensor_from_rig_scale;
  size_t observation_count;
  double preparation_seconds;
  double optimization_seconds;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

py::array_t<float> MatrixArray(std::vector<float>&& values,
                               const size_t rows,
                               const size_t columns) {
  auto* storage = new std::vector<float>(std::move(values));
  py::capsule owner(storage, [](void* pointer) {
    delete static_cast<std::vector<float>*>(pointer);
  });
  return py::array_t<float>(
      {static_cast<ssize_t>(rows), static_cast<ssize_t>(columns)},
      {static_cast<ssize_t>(columns * sizeof(float)),
       static_cast<ssize_t>(sizeof(float))},
      storage->data(),
      owner);
}

py::array_t<uint32_t> IndexArray(std::vector<uint32_t>&& values) {
  auto* storage = new std::vector<uint32_t>(std::move(values));
  py::capsule owner(storage, [](void* pointer) {
    delete static_cast<std::vector<uint32_t>*>(pointer);
  });
  return py::array_t<uint32_t>(storage->size(), storage->data(), owner);
}

std::vector<CasparRowTrackSource> NativeRowTrackSources(
    const std::vector<const PyCasparRowTrackSource*>& sources) {
  std::vector<CasparRowTrackSource> native;
  native.reserve(sources.size());
  for (const PyCasparRowTrackSource* source : sources) {
    THROW_CHECK_EQ(source->points.ndim(), 2);
    THROW_CHECK_EQ(source->points.shape(1), 3);
    THROW_CHECK_EQ(source->source_point_indices.ndim(), 1);
    THROW_CHECK_EQ(source->observation_offsets.ndim(), 1);
    THROW_CHECK_EQ(source->observation_image_indices.ndim(), 1);
    THROW_CHECK_EQ(source->observation_xy.ndim(), 2);
    THROW_CHECK_EQ(source->observation_xy.shape(1), 2);
    THROW_CHECK_EQ(source->duplicate_observation_indices.ndim(), 1);
    THROW_CHECK_EQ(source->image_rows.ndim(), 1);
    THROW_CHECK_GT(source->observation_offsets.shape(0), 0);
    const size_t num_tracks = source->observation_offsets.shape(0) - 1;
    THROW_CHECK_EQ(source->source_point_indices.shape(0), num_tracks);
    THROW_CHECK_EQ(source->observation_image_indices.shape(0),
                   source->observation_xy.shape(0));
    THROW_CHECK_EQ(source->observation_offsets.data()[num_tracks],
                   source->observation_xy.shape(0));
    native.push_back(CasparRowTrackSource{
        source->points.data(),
        source->points.strides(0) / static_cast<ssize_t>(sizeof(float)),
        source->points.strides(1) / static_cast<ssize_t>(sizeof(float)),
        source->source_point_indices.data(),
        source->observation_offsets.data(),
        source->observation_image_indices.data(),
        source->observation_xy.data(),
        source->duplicate_observation_indices.data(),
        source->image_rows.data(),
        num_tracks,
        static_cast<size_t>(source->duplicate_observation_indices.shape(0)),
    });
  }
  return native;
}

void CheckRowRigArrays(const std::vector<Rigid3d>& rigs_from_world,
                       const Uint32Array& image_frame_indices,
                       const Uint32Array& image_sensor_indices,
                       const std::vector<Rigid3d>& sensors_from_rig,
                       const FloatArray& sensor_calibrations) {
  THROW_CHECK_GE(rigs_from_world.size(), 2);
  THROW_CHECK_EQ(image_frame_indices.ndim(), 1);
  THROW_CHECK_GT(image_frame_indices.shape(0), 0);
  THROW_CHECK_EQ(image_sensor_indices.ndim(), 1);
  THROW_CHECK_EQ(image_sensor_indices.shape(0), image_frame_indices.shape(0));
  THROW_CHECK_GT(sensors_from_rig.size(), 0);
  THROW_CHECK_EQ(sensor_calibrations.ndim(), 2);
  THROW_CHECK_EQ(sensor_calibrations.shape(0), sensors_from_rig.size());
  THROW_CHECK_EQ(sensor_calibrations.shape(1), 4);
}

PyCasparRowSectionResult RefineRowSection(
    const std::vector<const PyCasparRowTrackSource*>& sources,
    const Uint32Array& point_track_offsets,
    const Uint32Array& point_track_indices,
    const Uint32Array& eligible_row_point_indices,
    const FloatArray& eligible_points,
    const std::vector<Rigid3d>& rigs_from_world,
    const Uint32Array& image_frame_indices,
    const Uint32Array& image_sensor_indices,
    const std::vector<Rigid3d>& sensors_from_rig,
    const FloatArray& sensor_calibrations,
    const BoolArray& active_frame_mask,
    const BoolArray& active_source_mask,
    const CasparBundleAdjustmentOptions& options) {
  THROW_CHECK_GT(sources.size(), 0);
  const std::vector<CasparRowTrackSource> native_sources =
      NativeRowTrackSources(sources);
  THROW_CHECK_EQ(point_track_offsets.ndim(), 1);
  THROW_CHECK_GT(point_track_offsets.shape(0), 1);
  THROW_CHECK_EQ(point_track_indices.ndim(), 1);
  THROW_CHECK_EQ(eligible_row_point_indices.ndim(), 1);
  THROW_CHECK_GT(eligible_row_point_indices.shape(0), 0);
  THROW_CHECK_EQ(eligible_points.ndim(), 2);
  THROW_CHECK_EQ(eligible_points.shape(0), eligible_row_point_indices.shape(0));
  THROW_CHECK_EQ(eligible_points.shape(1), 3);
  CheckRowRigArrays(rigs_from_world,
                    image_frame_indices,
                    image_sensor_indices,
                    sensors_from_rig,
                    sensor_calibrations);
  THROW_CHECK_EQ(active_frame_mask.ndim(), 1);
  THROW_CHECK_EQ(active_frame_mask.shape(0), rigs_from_world.size());
  THROW_CHECK_EQ(active_source_mask.ndim(), 1);
  THROW_CHECK_EQ(active_source_mask.shape(0), sources.size());

  CasparRowSectionResult result;
  {
    py::gil_scoped_release release;
    result = RefineRowSectionCaspar(native_sources,
                                    point_track_offsets.data(),
                                    point_track_indices.data(),
                                    eligible_row_point_indices.data(),
                                    eligible_points.data(),
                                    eligible_row_point_indices.shape(0),
                                    rigs_from_world,
                                    image_frame_indices.data(),
                                    image_sensor_indices.data(),
                                    image_frame_indices.shape(0),
                                    sensors_from_rig,
                                    sensor_calibrations.data(),
                                    active_frame_mask.data(),
                                    active_source_mask.data(),
                                    options);
  }
  const size_t num_points = result.row_point_indices.size();
  return {
      std::move(result.rigs_from_world),
      IndexArray(std::move(result.row_point_indices)),
      MatrixArray(std::move(result.points), num_points, 3),
      result.observation_count,
      result.active_observation_count,
      result.preparation_seconds,
      result.optimization_seconds,
      std::move(result.summary),
  };
}

PyCasparRowBundleResult OptimizeRow(
    const std::vector<const PyCasparRowTrackSource*>& sources,
    const Uint32Array& point_track_offsets,
    const Uint32Array& point_track_indices,
    const Uint32Array& point_observation_counts,
    const Uint32Array& selected_row_point_indices,
    const std::vector<Rigid3d>& rigs_from_world,
    const Uint32Array& image_frame_indices,
    const Uint32Array& image_sensor_indices,
    const std::vector<Rigid3d>& sensors_from_rig,
    const FloatArray& sensor_calibrations,
    const Uint32Array& prior_frame_indices,
    const FloatArray& prior_positions,
    const FloatArray& prior_sqrt_information,
    const CasparBundleAdjustmentOptions& options) {
  THROW_CHECK_GT(sources.size(), 0);
  const std::vector<CasparRowTrackSource> native_sources =
      NativeRowTrackSources(sources);
  THROW_CHECK_EQ(point_track_offsets.ndim(), 1);
  THROW_CHECK_GT(point_track_offsets.shape(0), 1);
  THROW_CHECK_EQ(point_track_indices.ndim(), 1);
  THROW_CHECK_EQ(point_observation_counts.ndim(), 1);
  THROW_CHECK_EQ(point_observation_counts.shape(0),
                 point_track_offsets.shape(0) - 1);
  THROW_CHECK_EQ(selected_row_point_indices.ndim(), 1);
  THROW_CHECK_GT(selected_row_point_indices.shape(0), 0);
  CheckRowRigArrays(rigs_from_world,
                    image_frame_indices,
                    image_sensor_indices,
                    sensors_from_rig,
                    sensor_calibrations);
  THROW_CHECK_EQ(prior_frame_indices.ndim(), 1);
  THROW_CHECK_EQ(prior_positions.ndim(), 2);
  THROW_CHECK_EQ(prior_positions.shape(1), 3);
  THROW_CHECK_EQ(prior_sqrt_information.ndim(), 3);
  THROW_CHECK_EQ(prior_sqrt_information.shape(1), 3);
  THROW_CHECK_EQ(prior_sqrt_information.shape(2), 3);
  const size_t num_priors = prior_frame_indices.shape(0);
  THROW_CHECK_GE(num_priors, 3);
  THROW_CHECK_EQ(prior_positions.shape(0), num_priors);
  THROW_CHECK_EQ(prior_sqrt_information.shape(0), num_priors);

  CasparRowBundleResult result;
  {
    py::gil_scoped_release release;
    result = OptimizeRowCaspar(native_sources,
                               point_track_offsets.data(),
                               point_track_indices.data(),
                               point_observation_counts.data(),
                               selected_row_point_indices.data(),
                               selected_row_point_indices.shape(0),
                               rigs_from_world,
                               image_frame_indices.data(),
                               image_sensor_indices.data(),
                               image_frame_indices.shape(0),
                               sensors_from_rig,
                               sensor_calibrations.data(),
                               prior_frame_indices.data(),
                               prior_positions.data(),
                               prior_sqrt_information.data(),
                               num_priors,
                               options);
  }
  const size_t num_points = result.row_point_indices.size();
  return {
      std::move(result.rigs_from_world),
      IndexArray(std::move(result.row_point_indices)),
      MatrixArray(std::move(result.points), num_points, 3),
      result.sensor_from_rig_scale,
      result.observation_count,
      result.preparation_seconds,
      result.optimization_seconds,
      std::move(result.summary),
  };
}

PyCasparRowPointRefinementResult RefineRowPoints(
    const std::vector<const PyCasparRowPointRefinementSource*>& sources,
    const Uint32Array& point_track_offsets,
    const Uint32Array& point_track_indices,
    const Uint32Array& point_observation_counts,
    const FloatArray& image_from_world,
    const Uint32Array& initialized_row_point_indices,
    const FloatArray& initialized_points,
    const size_t maximum_chunk_points,
    const size_t maximum_chunk_observations,
    const CasparRowPointRefinementOptions& options) {
  std::vector<const PyCasparRowTrackSource*> track_sources;
  track_sources.reserve(sources.size());
  for (const PyCasparRowPointRefinementSource* source : sources) {
    track_sources.push_back(source->tracks);
  }
  const std::vector<CasparRowTrackSource> native_track_sources =
      NativeRowTrackSources(track_sources);
  std::vector<CasparRowPointRefinementSource> native_refinement_sources;
  native_refinement_sources.reserve(sources.size());
  for (const PyCasparRowPointRefinementSource* source : sources) {
    THROW_CHECK_EQ(source->colors.ndim(), 2);
    THROW_CHECK_EQ(source->colors.shape(1), 3);
    THROW_CHECK_EQ(source->colors.shape(0), source->tracks->points.shape(0));
    THROW_CHECK_EQ(source->solved_world_from_source_world.ndim(), 3);
    THROW_CHECK_EQ(source->solved_world_from_source_world.shape(0),
                   source->tracks->image_rows.shape(0));
    THROW_CHECK_EQ(source->solved_world_from_source_world.shape(1), 4);
    THROW_CHECK_EQ(source->solved_world_from_source_world.shape(2), 4);
    native_refinement_sources.push_back(CasparRowPointRefinementSource{
        source->colors.data(),
        source->colors.strides(0) / static_cast<ssize_t>(sizeof(float)),
        source->colors.strides(1) / static_cast<ssize_t>(sizeof(float)),
        source->solved_world_from_source_world.data(),
    });
  }

  const size_t num_points = point_observation_counts.shape(0);
  CasparRowPointRefinementResult result;
  {
    py::gil_scoped_release release;
    result = RefineRowPointsCaspar(native_track_sources,
                                   native_refinement_sources,
                                   point_track_offsets.data(),
                                   point_track_indices.data(),
                                   point_observation_counts.data(),
                                   num_points,
                                   image_from_world.data(),
                                   image_from_world.shape(0),
                                   initialized_row_point_indices.data(),
                                   initialized_points.data(),
                                   initialized_row_point_indices.shape(0),
                                   maximum_chunk_points,
                                   maximum_chunk_observations,
                                   options);
  }
  return PyCasparRowPointRefinementResult{
      MatrixArray(std::move(result.points), num_points, 3),
      MatrixArray(std::move(result.colors), num_points, 3),
      result.chunk_count,
      result.iteration_count,
      result.maximum_iterations_run,
      result.runtime_seconds,
      result.initial_score,
      result.final_score,
      result.preparation_seconds,
      result.packing_seconds,
      result.optimization_seconds,
      result.validation_seconds,
      result.reprojection,
  };
}
#endif

}  // namespace

void BindRowPointRefinement(py::module& m) {
#ifdef CASPAR_ENABLED
  using RowOptions = CasparRowPointRefinementOptions;
  auto PyCasparRowPointRefinementOptions =
      py::classh<RowOptions, CasparPointRefinementOptions>(
          m, "CasparRowPointRefinementOptions")
          .def(py::init<>())
          .def_readwrite("validate_reprojection",
                         &RowOptions::validate_reprojection);
  MakeDataclass(PyCasparRowPointRefinementOptions);

  using ReprojectionSummary = CasparReprojectionErrorSummary;
  py::classh<ReprojectionSummary>(m, "CasparReprojectionErrorSummary")
      .def_readonly("point_count", &ReprojectionSummary::point_count)
      .def_readonly("observation_count",
                    &ReprojectionSummary::observation_count)
      .def_readonly("mean_pixels", &ReprojectionSummary::mean_pixels)
      .def_readonly("median_pixels", &ReprojectionSummary::median_pixels)
      .def_readonly("p95_pixels", &ReprojectionSummary::p95_pixels);

  py::classh<PyCasparRowPointRefinementResult>(m,
                                               "CasparRowPointRefinementResult")
      .def_readonly("points", &PyCasparRowPointRefinementResult::points)
      .def_readonly("colors", &PyCasparRowPointRefinementResult::colors)
      .def_readonly("chunk_count",
                    &PyCasparRowPointRefinementResult::chunk_count)
      .def_readonly("iteration_count",
                    &PyCasparRowPointRefinementResult::iteration_count)
      .def_readonly("maximum_iterations_run",
                    &PyCasparRowPointRefinementResult::maximum_iterations_run)
      .def_readonly("runtime_seconds",
                    &PyCasparRowPointRefinementResult::runtime_seconds)
      .def_readonly("initial_score",
                    &PyCasparRowPointRefinementResult::initial_score)
      .def_readonly("final_score",
                    &PyCasparRowPointRefinementResult::final_score)
      .def_readonly("preparation_seconds",
                    &PyCasparRowPointRefinementResult::preparation_seconds)
      .def_readonly("packing_seconds",
                    &PyCasparRowPointRefinementResult::packing_seconds)
      .def_readonly("optimization_seconds",
                    &PyCasparRowPointRefinementResult::optimization_seconds)
      .def_readonly("validation_seconds",
                    &PyCasparRowPointRefinementResult::validation_seconds)
      .def_readonly("reprojection",
                    &PyCasparRowPointRefinementResult::reprojection);

  py::classh<PyCasparRowTrackSource>(m, "CasparRowTrackSource")
      .def(py::init<StridedFloatArray,
                    Int64Array,
                    Int64Array,
                    Int32Array,
                    FloatArray,
                    Uint32Array,
                    Uint32Array>(),
           "points"_a.noconvert(),
           "source_point_indices"_a.noconvert(),
           "observation_offsets"_a.noconvert(),
           "observation_image_indices"_a.noconvert(),
           "observation_xy"_a.noconvert(),
           "duplicate_observation_indices"_a.noconvert(),
           "image_rows"_a.noconvert())
      .def_readonly("points", &PyCasparRowTrackSource::points)
      .def_readonly("source_point_indices",
                    &PyCasparRowTrackSource::source_point_indices)
      .def_readonly("observation_offsets",
                    &PyCasparRowTrackSource::observation_offsets)
      .def_readonly("observation_image_indices",
                    &PyCasparRowTrackSource::observation_image_indices)
      .def_readonly("observation_xy", &PyCasparRowTrackSource::observation_xy)
      .def_readonly("duplicate_observation_indices",
                    &PyCasparRowTrackSource::duplicate_observation_indices)
      .def_readonly("image_rows", &PyCasparRowTrackSource::image_rows);

  py::classh<PyCasparRowPointRefinementSource>(m,
                                               "CasparRowPointRefinementSource")
      .def(py::init<const PyCasparRowTrackSource*,
                    StridedFloatArray,
                    DoubleArray>(),
           "tracks"_a,
           "colors"_a.noconvert(),
           "solved_world_from_source_world"_a.noconvert(),
           py::keep_alive<1, 2>())
      .def_readonly("tracks", &PyCasparRowPointRefinementSource::tracks)
      .def_readonly("colors", &PyCasparRowPointRefinementSource::colors)
      .def_readonly(
          "solved_world_from_source_world",
          &PyCasparRowPointRefinementSource::solved_world_from_source_world);

  py::classh<PyCasparRowSectionResult>(m, "CasparRowSectionResult")
      .def_readonly("rigs_from_world",
                    &PyCasparRowSectionResult::rigs_from_world)
      .def_readonly("row_point_indices",
                    &PyCasparRowSectionResult::row_point_indices)
      .def_readonly("points", &PyCasparRowSectionResult::points)
      .def_readonly("observation_count",
                    &PyCasparRowSectionResult::observation_count)
      .def_readonly("active_observation_count",
                    &PyCasparRowSectionResult::active_observation_count)
      .def_readonly("preparation_seconds",
                    &PyCasparRowSectionResult::preparation_seconds)
      .def_readonly("optimization_seconds",
                    &PyCasparRowSectionResult::optimization_seconds)
      .def_readonly("summary", &PyCasparRowSectionResult::summary);

  py::classh<PyCasparRowBundleResult>(m, "CasparRowBundleResult")
      .def_readonly("rigs_from_world",
                    &PyCasparRowBundleResult::rigs_from_world)
      .def_readonly("row_point_indices",
                    &PyCasparRowBundleResult::row_point_indices)
      .def_readonly("points", &PyCasparRowBundleResult::points)
      .def_readonly("sensor_from_rig_scale",
                    &PyCasparRowBundleResult::sensor_from_rig_scale)
      .def_readonly("observation_count",
                    &PyCasparRowBundleResult::observation_count)
      .def_readonly("preparation_seconds",
                    &PyCasparRowBundleResult::preparation_seconds)
      .def_readonly("optimization_seconds",
                    &PyCasparRowBundleResult::optimization_seconds)
      .def_readonly("summary", &PyCasparRowBundleResult::summary);

  m.def("caspar_refine_row_points",
        RefineRowPoints,
        "sources"_a,
        "point_track_offsets"_a.noconvert(),
        "point_track_indices"_a.noconvert(),
        "point_observation_counts"_a.noconvert(),
        "image_from_world"_a.noconvert(),
        "initialized_row_point_indices"_a.noconvert(),
        "initialized_points"_a.noconvert(),
        "maximum_chunk_points"_a,
        "maximum_chunk_observations"_a,
        "options"_a = CasparRowPointRefinementOptions(),
        "Refine complete row points from canonical point-track CSR using "
        "fixed pinhole cameras and CASPAR.");
  m.def("caspar_refine_row_section",
        RefineRowSection,
        "sources"_a,
        "point_track_offsets"_a.noconvert(),
        "point_track_indices"_a.noconvert(),
        "eligible_row_point_indices"_a.noconvert(),
        "eligible_points"_a.noconvert(),
        "rigs_from_world"_a,
        "image_frame_indices"_a.noconvert(),
        "image_sensor_indices"_a.noconvert(),
        "sensors_from_rig"_a,
        "sensor_calibrations"_a.noconvert(),
        "active_frame_mask"_a.noconvert(),
        "active_source_mask"_a.noconvert(),
        "options"_a = CasparBundleAdjustmentOptions(),
        "Jointly refine a row section directly from canonical source tracks "
        "and point-track CSR. Inactive frame poses, sensor calibration, and "
        "the established global scale remain exact.");
  m.def("caspar_optimize_row",
        OptimizeRow,
        "sources"_a,
        "point_track_offsets"_a.noconvert(),
        "point_track_indices"_a.noconvert(),
        "point_observation_counts"_a.noconvert(),
        "selected_row_point_indices"_a.noconvert(),
        "rigs_from_world"_a,
        "image_frame_indices"_a.noconvert(),
        "image_sensor_indices"_a.noconvert(),
        "sensors_from_rig"_a,
        "sensor_calibrations"_a.noconvert(),
        "prior_frame_indices"_a.noconvert(),
        "prior_positions"_a.noconvert(),
        "prior_sqrt_information"_a.noconvert(),
        "options"_a = CasparBundleAdjustmentOptions(),
        "Jointly optimize a complete row directly from canonical source "
        "tracks and point-track CSR. Rig poses, selected points, and one "
        "shared sensor-translation scale are variable; sensor rotations and "
        "intrinsics remain fixed.");
#endif
}
