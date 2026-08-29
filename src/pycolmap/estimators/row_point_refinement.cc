#ifdef CASPAR_ENABLED
#include "colmap/estimators/row_point_refinement_caspar.h"
#endif

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
using FloatArray = py::array_t<float, py::array::c_style>;
using Int32Array = py::array_t<int32_t, py::array::c_style>;
using Int64Array = py::array_t<int64_t, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;

struct PyCasparRowPointRefinementSource {
  PyCasparRowPointRefinementSource(FloatArray points,
                                   FloatArray colors,
                                   Int64Array source_point_indices,
                                   Int64Array observation_offsets,
                                   Int32Array observation_image_indices,
                                   FloatArray observation_xy,
                                   Uint32Array duplicate_observation_indices,
                                   Uint32Array solved_image_rows,
                                   DoubleArray solved_world_from_source_world)
      : points(std::move(points)),
        colors(std::move(colors)),
        source_point_indices(std::move(source_point_indices)),
        observation_offsets(std::move(observation_offsets)),
        observation_image_indices(std::move(observation_image_indices)),
        observation_xy(std::move(observation_xy)),
        duplicate_observation_indices(std::move(duplicate_observation_indices)),
        solved_image_rows(std::move(solved_image_rows)),
        solved_world_from_source_world(
            std::move(solved_world_from_source_world)) {}

  FloatArray points;
  FloatArray colors;
  Int64Array source_point_indices;
  Int64Array observation_offsets;
  Int32Array observation_image_indices;
  FloatArray observation_xy;
  Uint32Array duplicate_observation_indices;
  Uint32Array solved_image_rows;
  DoubleArray solved_world_from_source_world;
};

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
    const CasparPointRefinementOptions& options) {
  std::vector<CasparRowPointRefinementSource> native_sources;
  native_sources.reserve(sources.size());
  for (const PyCasparRowPointRefinementSource* source : sources) {
    native_sources.push_back(CasparRowPointRefinementSource{
        source->points.data(),
        source->colors.data(),
        source->source_point_indices.data(),
        static_cast<size_t>(source->observation_offsets.shape(0) - 1),
        source->observation_offsets.data(),
        source->observation_image_indices.data(),
        source->observation_xy.data(),
        source->duplicate_observation_indices.data(),
        static_cast<size_t>(source->duplicate_observation_indices.shape(0)),
        source->solved_image_rows.data(),
        source->solved_world_from_source_world.data(),
    });
  }

  const size_t num_points = point_observation_counts.shape(0);
  CasparRowPointRefinementResult result;
  {
    py::gil_scoped_release release;
    result = RefineRowPointsCaspar(native_sources,
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
  };
}
#endif

}  // namespace

void BindRowPointRefinement(py::module& m) {
#ifdef CASPAR_ENABLED
  py::classh<PyCasparRowPointRefinementSource>(m,
                                               "CasparRowPointRefinementSource")
      .def(py::init<FloatArray,
                    FloatArray,
                    Int64Array,
                    Int64Array,
                    Int32Array,
                    FloatArray,
                    Uint32Array,
                    Uint32Array,
                    DoubleArray>(),
           "points"_a,
           "colors"_a,
           "source_point_indices"_a,
           "observation_offsets"_a,
           "observation_image_indices"_a,
           "observation_xy"_a,
           "duplicate_observation_indices"_a,
           "solved_image_rows"_a,
           "solved_world_from_source_world"_a);

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
                    &PyCasparRowPointRefinementResult::optimization_seconds);

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
        "options"_a = CasparPointRefinementOptions(),
        "Refine complete row points from canonical point-track CSR using "
        "fixed pinhole cameras and CASPAR.");
#endif
}
