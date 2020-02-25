//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_utils_is_aligned_hpp__
#define __eigenpy_utils_is_aligned_hpp__

namespace eigenpy
{
  inline bool is_aligned(void * ptr, std::size_t alignment)
  {
    return (reinterpret_cast<std::size_t>(ptr) & (alignment - 1)) == 0;
  }
}

#endif
