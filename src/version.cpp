//
// Copyright (c) 2019 INRIA
//

#include "eigenpy/version.hpp"

#include <sstream>

#include "eigenpy/config.hpp"

namespace eigenpy {

std::string printVersion(const std::string& delimiter) {
  std::ostringstream oss;
  oss << EIGENPY_MAJOR_VERSION << delimiter << EIGENPY_MINOR_VERSION
      << delimiter << EIGENPY_PATCH_VERSION;
  return oss.str();
}

bool checkVersionAtLeast(unsigned int major_version, unsigned int minor_version,
                         unsigned int patch_version) {
  return EIGENPY_VERSION_AT_LEAST(major_version, minor_version, patch_version);
}

}  // namespace eigenpy
