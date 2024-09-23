#include <boost/python.hpp>

namespace eigenpy {

struct EmptyPythonVisitor
    : public ::boost::python::def_visitor<EmptyPythonVisitor> {
  template <class classT>
  void visit(classT &) const {}
};

}  // namespace eigenpy
