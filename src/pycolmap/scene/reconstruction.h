#pragma once

#include "colmap/scene/reconstruction.h"

#include <pybind11/pybind11.h>

pybind11::dict TrackArrays(const colmap::Reconstruction& reconstruction);
