#include "pycolmap/feature/fixed_rig_matching.h"

using namespace colmap;

void BindFixedRigMatchingOptions(
    pybind11::classh<FeatureMatchingOptions>& options_class) {
  options_class.def_readwrite(
      "use_fixed_rig_geometry",
      &FeatureMatchingOptions::use_fixed_rig_geometry,
      "Whether to use fixed rig geometry for guided matching between images "
      "in the same frame.");
}
