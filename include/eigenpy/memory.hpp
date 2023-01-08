/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#ifndef __eigenpy_memory_hpp__
#define __eigenpy_memory_hpp__

#include "eigenpy/fwd.hpp"

/**
 * This section contains a convenience MACRO which allows an easy specialization
 * of Boost Python Object allocator for struct data types containing Eigen
 * objects and requiring strict alignment.
 */
#define EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(...)         \
  namespace boost {                                                 \
  namespace python {                                                \
  namespace objects {                                               \
  template <>                                                       \
  struct instance<value_holder<__VA_ARGS__> >                       \
      : ::eigenpy::aligned_instance<value_holder<__VA_ARGS__> > {}; \
  }                                                                 \
  }                                                                 \
  }

#endif  // __eigenpy_memory_hpp__
