//
// Copyright (c) 2014-2024 CNRS INRIA
//

#ifndef __eigenpy_scalar_conversion_hpp__
#define __eigenpy_scalar_conversion_hpp__

#include "eigenpy/config.hpp"
#include <boost/numeric/conversion/conversion_traits.hpp>

namespace eigenpy {
template <typename Source, typename Target>
struct FromTypeToType
    : public boost::numeric::conversion_traits<Source, Target>::subranged {};

}  // namespace eigenpy

#endif  // __eigenpy_scalar_conversion_hpp__
