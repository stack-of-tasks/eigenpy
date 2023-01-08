/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include "eigenpy/fwd.hpp"

EIGENPY_DEPRECATED_FILE(
    "This header file is now useless and should not be included anymore.")

#ifndef __eigenpy_memory_hpp__
#define __eigenpy_memory_hpp__

/**
 * This section contains a convenience MACRO which allows an easy specialization
 * of Boost Python Object allocator for struct data types containing Eigen
 * objects and requiring strict alignment.
 */
#define EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(...)                  \
  EIGENPY_DEPRECATED_MACRO(EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(), \
                           "it is no more needed.")

#endif  // __eigenpy_memory_hpp__
