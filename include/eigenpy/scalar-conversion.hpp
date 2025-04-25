//
// Copyright (c) 2014-2024 CNRS INRIA
//

#ifndef __eigenpy_scalar_conversion_hpp__
#define __eigenpy_scalar_conversion_hpp__

#include "eigenpy/config.hpp"
#include <boost/numeric/conversion/conversion_traits.hpp>
#include <complex>

namespace eigenpy {

template <typename Source, typename Target>
struct FromTypeToType
    : public boost::mpl::if_c<std::is_same<Source, Target>::value,
                              std::true_type,
                              typename boost::numeric::conversion_traits<
                                  Source, Target>::subranged>::type {};

/// FromTypeToType specialization to manage std::complex
template <typename ScalarSource, typename ScalarTarget>
struct FromTypeToType<std::complex<ScalarSource>, std::complex<ScalarTarget>>
    : public boost::mpl::if_c<
          std::is_same<ScalarSource, ScalarTarget>::value, std::true_type,
          typename boost::numeric::conversion_traits<
              ScalarSource, ScalarTarget>::subranged>::type {};

}  // namespace eigenpy

#endif  // __eigenpy_scalar_conversion_hpp__
