#include <boost/numpy.hpp>

#include "eigenpy/fwd.hpp"

namespace boopy {
namespace bpn = boost::numpy;
namespace bp = boost::python;

struct Eigenvec_to_python_matrix {
  static PyObject* convert(Eigen::VectorXd const& v) {
    Py_intptr_t shape[1] = {v.size()};
    bpn::matrix result(bpn::zeros(1, shape, bpn::dtype::get_builtin<double>()));
    std::copy(v.data(), v.data() + v.size(),
              reinterpret_cast<double*>(result.get_data()));
    return bp::incref(result.ptr());
  }
};

struct Eigenvec_from_python_array {
  Eigenvec_from_python_array() {
    bp::converter::registry ::push_back(&convertible, &construct,
                                        bp::type_id<Eigen::VectorXd>());
  }

  // Determine if obj_ptr can be converted in a Eigenvec
  static void* convertible(PyObject* obj_ptr) {
    try {
      bp::object obj(bp::handle<>(bp::borrowed(obj_ptr)));
      std::auto_ptr<bpn::ndarray> array(new bpn::ndarray(bpn::from_object(
          obj, bpn::dtype::get_builtin<double>(), bpn::ndarray::V_CONTIGUOUS)));

      if ((array->get_nd() == 1) ||
          ((array->get_nd() == 2) && (array->get_shape()[1] == 1)))
        return array.release();
      else
        return 0;
    } catch (bp::error_already_set& err) {
      bp::handle_exception();
      return 0;
    }
  }

  // Convert obj_ptr into a Eigenvec
  static void construct(PyObject*,
                        bp::converter::rvalue_from_python_stage1_data* memory) {
    // Recover the pointer created in <convertible>
    std::auto_ptr<bpn::ndarray> array(
        reinterpret_cast<bpn::ndarray*>(memory->convertible));
    const int nrow = array->get_shape()[0];
    std::cout << "nrow = " << nrow << std::endl;

    // Get the memory where to create the vector
    void* storage =
        ((bp::converter::rvalue_from_python_storage<Eigen::VectorXd>*)memory)
            ->storage.bytes;

    // Create the vector
    Eigen::VectorXd& res = *new (storage) Eigen::VectorXd(nrow);

    // Copy the data
    double* data = (double*)array->get_data();
    for (int i = 0; i < nrow; ++i) res[i] = data[i];

    // Stash the memory chunk pointer for later use by boost.python
    memory->convertible = storage;
  }
};
}  // namespace boopy

Eigen::VectorXd test() {
  Eigen::VectorXd v = Eigen::VectorXd::Random(5);
  std::cout << v.transpose() << std::endl;
  return v;
}

void test2(Eigen::VectorXd v) {
  std::cout << "test2: dim = " << v.size() << " ||| v[0] = " << v[0]
            << std::endl;
}

BOOST_PYTHON_MODULE(libeigen) {
  namespace bpn = boost::numpy;
  namespace bp = boost::python;

  bpn::initialize();
  bp::to_python_converter<Eigen::VectorXd, boopy::Eigenvec_to_python_matrix>();
  boopy::Eigenvec_from_python_array();

  bp::def("test", test);
  bp::def("test2", test2);
}
