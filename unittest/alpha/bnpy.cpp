/* Simple test using the boost::numpy interface: return an array and a matrix.
 */

#include "boost/numpy.hpp"
#include "eigenpy/fwd.hpp"

namespace bp = boost::python;
namespace bn = boost::numpy;

/* Return an dim-1 array with 5 elements. */
bn::ndarray array() {
  std::vector<double> v(5);
  v[0] = 56;
  Py_intptr_t shape[1] = {v.size()};
  bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<double>());
  std::copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));
  return result;
}

/* Return a dim-1 matrix with five elements. */
boost::python::object matrix() {
  std::vector<double> v(5);
  v[0] = 56;
  Py_intptr_t shape[1] = {v.size()};
  bn::matrix t(bn::zeros(1, shape, bn::dtype::get_builtin<double>()));
  std::copy(v.begin(), v.end(), reinterpret_cast<double*>(t.get_data()));

  return t;
}

BOOST_PYTHON_MODULE(libbnpy) {
  bn::initialize();
  bp::def("array", array);
  bp::def("matrix", matrix);
}
