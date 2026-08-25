#pragma once

#include "colmap/feature/matcher.h"

#include "pycolmap/pybind11_extension.h"

void BindFixedRigMatchingOptions(
    pybind11::classh<colmap::FeatureMatchingOptions>& options_class);
