/* Simple test with boost::python.
 * Declare and bind three function, returning char*, string, and Eigen::Vector.
 * The last function raises and error at runtime due to inadequate binding.
 */

#include <string>

#include "eigenpy/fwd.hpp"

char const* testchar() { return "Yay char!"; }

std::string teststr() { return "Yay str!"; }

Eigen::VectorXd testeigenvec() {
  Eigen::VectorXd v(555);
  return v;
}

BOOST_PYTHON_MODULE(libsimple) {
  using namespace boost::python;
  def("char", testchar);
  def("str", teststr);
  def("eigenvec", testeigenvec);
}
