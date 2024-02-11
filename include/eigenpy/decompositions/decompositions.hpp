/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_decompositions_decompositions_hpp__
#define __eigenpy_decompositions_decompositions_hpp__

#include "eigenpy/config.hpp"

namespace eigenpy {
void EIGENPY_DLLAPI exposeDecompositions();

#ifdef EIGENPY_WITH_CHOLMOD_SUPPORT
void EIGENPY_DLLAPI exposeCholmod();
#endif

}  // namespace eigenpy

#endif  // define __eigenpy_decompositions_decompositions_hpp__
