#include "header.hpp"

BOOST_PYTHON_MODULE(
    eigenpy_example_custom_numeric_type)  // this name must match the name of
                                          // the generated .so file.
{
  // see
  // https://stackoverflow.com/questions/6114462/how-to-override-the-automatically-created-docstring-data-for-boostpython
  // docstring_options d(true, true, false); // local_
  boost::python::docstring_options docopt;
  docopt.enable_all();
  docopt.disable_cpp_signatures();

  boost::python::object package = boost::python::scope();
  package.attr("__path__") = "eigenpy_example_custom_numeric_type";

  ExposeAll();
}

#define IMPLICITLY_CONVERTIBLE(T1, T2) \
  boost::python::implicitly_convertible<T1, T2>();



void ExposeAll(){
  eigenpy::enableEigenPy();

  ExposeReal();
  ExposeComplex();
}

void ExposeReal() {

  BoostNumberPythonVisitor<mpfr_float>::expose("MpfrFloat");


  using VecX = Eigen::Matrix<mpfr_float, Eigen::Dynamic, 1>;
  using MatXX = Eigen::Matrix<mpfr_float, Eigen::Dynamic, Eigen::Dynamic>;

  eigenpy::enableEigenPySpecific<MatXX>();
  eigenpy::enableEigenPySpecific<VecX>();
}


void ExposeComplex() {

  boost::python::class_<mpfr_complex>("MpfrComplex", "", bp::no_init)
      .def(BoostComplexPythonVisitor<mpfr_complex>());

  eigenpy::registerNewType<mpfr_complex>();
  eigenpy::registerUfunct_without_comparitors<mpfr_complex>();

  eigenpy::registerCast<long, mpfr_complex>(true);
  eigenpy::registerCast<int, mpfr_complex>(true);
  eigenpy::registerCast<int64_t, mpfr_complex>(true);

  IMPLICITLY_CONVERTIBLE(int, mpfr_complex);
  IMPLICITLY_CONVERTIBLE(long, mpfr_complex);
  IMPLICITLY_CONVERTIBLE(int64_t, mpfr_complex);

  using VecX = Eigen::Matrix<mpfr_complex, Eigen::Dynamic, 1>;
  using MatXX = Eigen::Matrix<mpfr_complex, Eigen::Dynamic, Eigen::Dynamic>;

  eigenpy::enableEigenPySpecific<MatXX>();
  eigenpy::enableEigenPySpecific<VecX>();
}

#undef IMPLICITLY_CONVERTIBLE
