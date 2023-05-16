///
/// Copyright 2023 CNRS, INRIA
///

#include "eigenpy/optional.hpp"

namespace eigenpy {
void exposeNoneType() {
  detail::NoneToPython<boost::none_t>::registration();
#ifdef EIGENPY_WITH_CXX17_SUPPORT
  detail::NoneToPython<std::nullopt_t>::registration();
#endif
}
}  // namespace eigenpy
