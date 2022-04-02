#include <numpy/arrayobject.h>

#include "eigenpy/fwd.hpp"

namespace boopy {
namespace bp = boost::python;

template <typename SCALAR>
struct NumpyEquivalentType {};
template <>
struct NumpyEquivalentType<double> {
  enum { type_code = NPY_DOUBLE };
};
template <>
struct NumpyEquivalentType<int> {
  enum { type_code = NPY_INT };
};
template <>
struct NumpyEquivalentType<float> {
  enum { type_code = NPY_FLOAT };
};

struct EigenMatrix_to_python_matrix {
  static PyObject* convert(Eigen::MatrixXd const& mat) {
    typedef Eigen::MatrixXd::Scalar T;

    npy_intp shape[2] = {mat.rows(), mat.cols()};
    PyArrayObject* pyArray = (PyArrayObject*)PyArray_SimpleNew(
        2, shape, NumpyEquivalentType<T>::type_code);

    T* pyData = (T*)PyArray_DATA(pyArray);
    for (int i = 0; i < mat.rows(); ++i)
      for (int j = 0; j < mat.cols(); ++j)
        pyData[i * mat.cols() + j] = mat(i, j);

    return ((PyObject*)pyArray);
  }
};

struct EigenMatrix_from_python_array {
  EigenMatrix_from_python_array() {
    bp::converter::registry ::push_back(&convertible, &construct,
                                        bp::type_id<Eigen::MatrixXd>());
  }

  // Determine if obj_ptr can be converted in a Eigenvec
  static void* convertible(PyObject* obj_ptr) {
    typedef Eigen::MatrixXd::Scalar T;

    if (!PyArray_Check(obj_ptr)) {
      return 0;
    }
    if (PyArray_NDIM(obj_ptr) > 2) {
      return 0;
    }
    if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<T>::type_code) {
      return 0;
    }
    int flags = PyArray_FLAGS(obj_ptr);
    if (!(flags & NPY_C_CONTIGUOUS)) {
      return 0;
    }
    if (!(flags & NPY_ALIGNED)) {
      return 0;
    }

    return obj_ptr;
  }

  // Convert obj_ptr into a Eigenvec
  static void construct(PyObject* pyObj,
                        bp::converter::rvalue_from_python_stage1_data* memory) {
    typedef Eigen::MatrixXd::Scalar T;
    using namespace Eigen;

    PyArrayObject* pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
    int ndims = PyArray_NDIM(pyArray);
    assert(ndims == 2);

    int dtype_size = (PyArray_DESCR(pyArray))->elsize;
    int s1 = PyArray_STRIDE(pyArray, 0);
    assert(s1 % dtype_size == 0);

    int R = MatrixXd::RowsAtCompileTime;
    int C = MatrixXd::ColsAtCompileTime;
    if (R == Eigen::Dynamic)
      R = PyArray_DIMS(pyArray)[0];
    else
      assert(PyArray_DIMS(pyArray)[0] == R);

    if (C == Eigen::Dynamic)
      C = PyArray_DIMS(pyArray)[1];
    else
      assert(PyArray_DIMS(pyArray)[1] == C);

    T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));

    void* storage =
        ((bp::converter::rvalue_from_python_storage<MatrixXd>*)(memory))
            ->storage.bytes;
    MatrixXd& mat = *new (storage) MatrixXd(R, C);
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < C; ++j) mat(i, j) = pyData[i * C + j];

    memory->convertible = storage;
  }
};

}  // namespace boopy

Eigen::MatrixXd test() {
  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(5, 5);
  std::cout << "EigenMAt = " << mat << std::endl;
  return mat;
}

void test2(Eigen::MatrixXd mat) {
  std::cout << "test2: dim = " << mat.rows() << " ||| m[0,0] = " << mat(0, 0)
            << std::endl;
}

BOOST_PYTHON_MODULE(libeigenc) {
  import_array();
  namespace bp = boost::python;
  bp::to_python_converter<Eigen::MatrixXd,
                          boopy::EigenMatrix_to_python_matrix>();
  boopy::EigenMatrix_from_python_array();

  bp::def("test", test);
  bp::def("test2", test2);
}
