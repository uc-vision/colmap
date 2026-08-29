#include "pycolmap/pybind11_extension.h"

#include <cstdint>

#include <Eigen/Core>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;
using UInt32Array = py::array_t<uint32_t, py::array::c_style>;

template <typename Scalar>
using PointMatrix =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <typename Scalar>
void AccumulateIndexedPoints(
    const py::array_t<Scalar, py::array::c_style>& point_positions,
    const UInt32Array& point_indices,
    DoubleArray point_sums,
    UInt32Array point_counts) {
  py::gil_scoped_release release;
  const Eigen::Map<const PointMatrix<Scalar>> positions(
      point_positions.data(), point_positions.shape(0), 3);
  Eigen::Map<PointMatrix<double>> sums(
      point_sums.mutable_data(), point_sums.shape(0), 3);
  uint32_t* counts = point_counts.mutable_data();
  const uint32_t* indices = point_indices.data();
  for (Eigen::Index row = 0; row < positions.rows(); ++row) {
    const uint32_t point = indices[row];
    sums.row(point) += positions.row(row).template cast<double>();
    ++counts[point];
  }
}

}  // namespace

void BindRowPointAccumulation(py::module& m) {
  m.def("accumulate_indexed_points",
        &AccumulateIndexedPoints<float>,
        "point_positions"_a,
        "point_indices"_a,
        "point_sums"_a,
        "point_counts"_a);
  m.def("accumulate_indexed_points",
        &AccumulateIndexedPoints<double>,
        "point_positions"_a,
        "point_indices"_a,
        "point_sums"_a,
        "point_counts"_a);
}
