#include "header.hpp"


// a placeholder for a library function that writes data into an existing matrix
template <typename T>
void set_to_ones(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>  M){
  for (Eigen::Index ii=0; ii<M.rows(); ++ii){
    for (Eigen::Index jj=0; jj<M.cols(); ++jj){
      M(ii,jj) = T(1);
    }
  }

}


template <typename T>
void set_all_entries_to_constant(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> M, T const& the_constant){

  for (Eigen::Index ii=0; ii<M.rows(); ++ii){
    for (Eigen::Index jj=0; jj<M.cols(); ++jj){
      M(ii,jj) = the_constant;
    }
  }
}


template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a_function_taking_both_a_scalar_and_a_vector(T const& scalar, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const& M)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result;
  set_all_entries_to_constant(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>{result}, scalar);
  return result;
}

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

void ExposeAll() {
  eigenpy::enableEigenPy();

  ExposeReal();
  ExposeComplex();

  // some free C++ functions that do stuff to Eigen::Matrix types.  useful to prove they work.
  boost::python::def("set_to_ones", &set_to_ones<mpfr_float>, "set an array to all ones");
  boost::python::def("set_to_ones", &set_to_ones<mpfr_complex>, "set an array to all ones");

  boost::python::def("set_all_entries_to_constant", &set_all_entries_to_constant<mpfr_float>, "set an array to all one value, from a given number of the same type");
  boost::python::def("set_all_entries_to_constant", &set_all_entries_to_constant<mpfr_complex>, "set an array to all one value, from a given number of the same type");
  

  boost::python::def("make_a_vector_in_cpp", &make_a_vector_in_cpp<mpfr_float>, "make a vector from c++");
  boost::python::def("make_a_vector_in_cpp", &make_a_vector_in_cpp<mpfr_complex>, "make a vector from c++");

  boost::python::def("a_function_taking_both_a_scalar_and_a_vector", &a_function_taking_both_a_scalar_and_a_vector<mpfr_float>, "do nothing, but accept both a scalar and a vector");
  boost::python::def("a_function_taking_both_a_scalar_and_a_vector", &a_function_taking_both_a_scalar_and_a_vector<mpfr_complex>, "do nothing, but accept both a scalar and a vector");


  // showing we can expose classes that do stuff with exposed types
  ExposeAClass();
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
