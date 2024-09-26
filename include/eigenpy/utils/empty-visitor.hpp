#ifndef __eigenpy_utils_empty_visitor_hpp__
#define __eigenpy_utils_empty_visitor_hpp__

#include <boost/python.hpp>

namespace eigenpy {

struct EmptyPythonVisitor
    : public ::boost::python::def_visitor<EmptyPythonVisitor> {
  template <class classT>
  void visit(classT &) const {}
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_empty_visitor_hpp__
