#include "colmap/estimators/fixed_rig_two_view_geometry.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

using FeaturePointsMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

void BindFixedRigTwoViewGeometryEstimator(py::module& m) {
  auto PyFixedRigMatchedPair =
      py::classh<FixedRigMatchedPair>(m, "FixedRigMatchedPair")
          .def(py::init<>())
          .def(py::init(
                   [](const image_t image_id1,
                      const image_t image_id2,
                      const Camera& camera1,
                      const Camera& camera2,
                      const Eigen::Ref<const FeatureMatchesMatrix>& matches,
                      const Eigen::Ref<const FeaturePointsMatrix>& points1,
                      const Eigen::Ref<const FeaturePointsMatrix>& points2) {
                     FixedRigMatchedPair pair;
                     pair.image_id1 = image_id1;
                     pair.image_id2 = image_id2;
                     pair.camera1 = camera1;
                     pair.camera2 = camera2;
                     pair.matches = MatchesFromMatrix(matches);
                     pair.points1.resize(points1.rows());
                     pair.points2.resize(points2.rows());
                     Eigen::Map<FeaturePointsMatrix>(
                         reinterpret_cast<double*>(pair.points1.data()),
                         points1.rows(),
                         2) = points1;
                     Eigen::Map<FeaturePointsMatrix>(
                         reinterpret_cast<double*>(pair.points2.data()),
                         points2.rows(),
                         2) = points2;
                     return pair;
                   }),
               "image_id1"_a,
               "image_id2"_a,
               "camera1"_a,
               "camera2"_a,
               "matches"_a,
               "points1"_a,
               "points2"_a)
          .def_readwrite("image_id1", &FixedRigMatchedPair::image_id1)
          .def_readwrite("image_id2", &FixedRigMatchedPair::image_id2)
          .def_readwrite("camera1", &FixedRigMatchedPair::camera1)
          .def_readwrite("camera2", &FixedRigMatchedPair::camera2)
          .def_property(
              "matches",
              [](const FixedRigMatchedPair& self) {
                return MatchesToMatrix(self.matches);
              },
              [](FixedRigMatchedPair& self,
                 const Eigen::Ref<const FeatureMatchesMatrix>& matches) {
                self.matches = MatchesFromMatrix(matches);
              })
          .def_readwrite("points1", &FixedRigMatchedPair::points1)
          .def_readwrite("points2", &FixedRigMatchedPair::points2);
  MakeDataclass(PyFixedRigMatchedPair);

  m.def(
      "estimate_fixed_rig_two_view_geometries",
      [](const Rig& rig1,
         const Rig& rig2,
         const std::vector<const FixedRigMatchedPair*>& pair_refs,
         const TwoViewGeometryOptions& options,
         const size_t max_num_ransac_matches) {
        py::gil_scoped_release release;
        std::vector<FixedRigMatchedPair> pairs;
        pairs.reserve(pair_refs.size());
        for (const FixedRigMatchedPair* pair : pair_refs) {
          pairs.push_back(*pair);
        }
        auto geometries = EstimateFixedRigTwoViewGeometries(
            rig1, rig2, pairs, options, max_num_ransac_matches);
        std::vector<std::tuple<image_t, image_t, TwoViewGeometry>> result;
        result.reserve(geometries.size());
        for (auto& [image_pair, geometry] : geometries) {
          result.emplace_back(
              image_pair.first, image_pair.second, std::move(geometry));
        }
        return result;
      },
      "rig1"_a,
      "rig2"_a,
      "pairs"_a,
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"),
      "max_num_ransac_matches"_a = 4096,
      "Estimate one fixed-rig relative pose without accessing a database.");
}
