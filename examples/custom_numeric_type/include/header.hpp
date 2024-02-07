

#pragma once

#ifndef EXAMPLE_CUSTOM_NUMERIC_TYPE
#define EXAMPLE_CUSTOM_NUMERIC_TYPE

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/user-type.hpp>
#include <eigenpy/ufunc.hpp>

#include <boost/multiprecision/mpc.hpp>

#include <boost/multiprecision/eigen.hpp>

namespace bmp = boost::multiprecision;

using mpfr_float =
    boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<0>,
                                  boost::multiprecision::et_off>;

using bmp::backends::mpc_complex_backend;
using mpfr_complex =
    bmp::number<mpc_complex_backend<0>,
                bmp::et_off>;  // T is a variable-precision complex number with
                               // expression templates turned on.

void ExposeAll();
void ExposeReal();
void ExposeComplex();

// this code derived from
// https://github.com/stack-of-tasks/eigenpy/issues/365
// where I asked about using custom types, and @jcarpent responded with a
// discussion of an application of this in Pinnochio, a library for rigid body
// dynamics.
namespace eigenpy {
namespace internal {

// a template specialization for complex numbers
// derived directly from the example for Pinnochio
template <>
struct getitem<mpfr_float> {
  static PyObject *run(void *data, void * /* arr */) {
    mpfr_float &mpfr_scalar = *static_cast<mpfr_float *>(data);
    auto &backend = mpfr_scalar.backend();

    if (backend.data()[0]._mpfr_d ==
        0)  // If the mpfr_scalar is not initialized, we have to init it.
    {
      mpfr_scalar = mpfr_float(0);
    }
    boost::python::object m(boost::ref(mpfr_scalar));
    Py_INCREF(m.ptr());
    return m.ptr();
  }
};

// a template specialization for complex numbers
// derived directly from the example for Pinnochio
template <>
struct getitem<mpfr_complex> {
  static PyObject *run(void *data, void * /* arr */) {
    mpfr_complex &mpfr_scalar = *static_cast<mpfr_complex *>(data);
    auto &backend = mpfr_scalar.backend();

    if (backend.data()[0].re->_mpfr_d == 0 ||
        backend.data()[0].im->_mpfr_d ==
            0)  // If the mpfr_scalar is not initialized, we have to init it.
    {
      mpfr_scalar = mpfr_complex(0);
    }
    boost::python::object m(boost::ref(mpfr_scalar));
    Py_INCREF(m.ptr());
    return m.ptr();
  }
};

}  // namespace internal

// i lifted this from EigenPy and adapted it, basically removing the calls for
// the comparitors.
template <typename Scalar>
void registerUfunct_without_comparitors() {
  const int type_code = Register::getTypeCode<Scalar>();

  PyObject *numpy_str;
#if PY_MAJOR_VERSION >= 3
  numpy_str = PyUnicode_FromString("numpy");
#else
  numpy_str = PyString_FromString("numpy");
#endif
  PyObject *numpy;
  numpy = PyImport_Import(numpy_str);
  Py_DECREF(numpy_str);

  import_ufunc();

  // Matrix multiply
  {
    int types[3] = {type_code, type_code, type_code};

    std::stringstream ss;
    ss << "return result of multiplying two matrices of ";
    ss << bp::type_info(typeid(Scalar)).name();
    PyUFuncObject *ufunc =
        (PyUFuncObject *)PyObject_GetAttrString(numpy, "matmul");
    if (!ufunc) {
      std::stringstream ss;
      ss << "Impossible to define matrix_multiply for given type "
         << bp::type_info(typeid(Scalar)).name() << std::endl;
      eigenpy::Exception(ss.str());
    }
    if (PyUFunc_RegisterLoopForType((PyUFuncObject *)ufunc, type_code,
                                    &internal::gufunc_matrix_multiply<Scalar>,
                                    types, 0) < 0) {
      std::stringstream ss;
      ss << "Impossible to register matrix_multiply for given type "
         << bp::type_info(typeid(Scalar)).name() << std::endl;
      eigenpy::Exception(ss.str());
    }

    Py_DECREF(ufunc);
  }

  // Binary operators
  EIGENPY_REGISTER_BINARY_UFUNC(add, type_code, Scalar, Scalar, Scalar);
  EIGENPY_REGISTER_BINARY_UFUNC(subtract, type_code, Scalar, Scalar, Scalar);
  EIGENPY_REGISTER_BINARY_UFUNC(multiply, type_code, Scalar, Scalar, Scalar);
  EIGENPY_REGISTER_BINARY_UFUNC(divide, type_code, Scalar, Scalar, Scalar);

  // Comparison operators
  EIGENPY_REGISTER_BINARY_UFUNC(equal, type_code, Scalar, Scalar, bool);
  EIGENPY_REGISTER_BINARY_UFUNC(not_equal, type_code, Scalar, Scalar, bool);

  // these are commented out because the comparisons are NOT defined for complex
  // types!!
  //  EIGENPY_REGISTER_BINARY_UFUNC(greater, type_code, Scalar, Scalar, bool);
  //  EIGENPY_REGISTER_BINARY_UFUNC(less, type_code, Scalar, Scalar, bool);
  //  EIGENPY_REGISTER_BINARY_UFUNC(greater_equal, type_code, Scalar, Scalar,
  //  bool); EIGENPY_REGISTER_BINARY_UFUNC(less_equal, type_code, Scalar,
  //  Scalar, bool);

  // Unary operators
  EIGENPY_REGISTER_UNARY_UFUNC(negative, type_code, Scalar, Scalar);

  Py_DECREF(numpy);
}

}  // namespace eigenpy

namespace bp = boost::python;

template <typename BoostNumber>
struct BoostNumberPythonVisitor : public boost::python::def_visitor<
                                      BoostNumberPythonVisitor<BoostNumber> > {
 public:
  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>("Default constructor.", bp::arg("self")))
        .def(bp::init<BoostNumber>("Copy constructor.",
                                   bp::args("self", "value")))
        //        .def(bp::init<bool>("Copy
        //        constructor.",bp::args("self","value")))
        //        .def(bp::init<float>("Copy
        //        constructor.",bp::args("self","value")))
        //        .def(bp::init<double>("Copy
        //        constructor.",bp::args("self","value")))
        //        .def(bp::init<int>("Copy
        //        constructor.",bp::args("self","value"))) .def(bp::init<long
        //        int>("Copy constructor.",bp::args("self","value")))
        //        .def(bp::init<unsigned int>("Copy
        //        constructor.",bp::args("self","value")))
        //        .def(bp::init<unsigned long int>("Copy
        //        constructor.",bp::args("self","value")))
        .def(bp::init<std::string>("Constructor from a string.",
                                   bp::args("self", "str_value")))

        .def(bp::self + bp::self)
        .def(bp::self += bp::self)
        .def(bp::self - bp::self)
        .def(bp::self -= bp::self)
        .def(bp::self * bp::self)
        .def(bp::self *= bp::self)
        .def(bp::self / bp::self)
        .def(bp::self /= bp::self)

        .def(bp::self < bp::self)
        .def(bp::self <= bp::self)
        .def(bp::self > bp::self)
        .def(bp::self >= bp::self)
        .def(bp::self == bp::self)
        .def(bp::self != bp::self)
        .def(bp::self_ns::pow(bp::self_ns::self, long()))

        .def("str", &BoostNumber::str,
             bp::args("self", "precision", "scientific"))

        .def("default_precision",
             static_cast<unsigned (*)()>(BoostNumber::default_precision),
             "Get the default precision of the class.")
        .def("default_precision",
             static_cast<void (*)(unsigned)>(BoostNumber::default_precision),
             bp::arg("digits10"), "Set the default precision of the class.")
        .staticmethod("default_precision")

        .def("precision",
             static_cast<unsigned (BoostNumber::*)() const>(
                 &BoostNumber::precision),
             bp::arg("self"), "Get the precision of this.")
        .def("precision",
             static_cast<void (BoostNumber::*)(unsigned)>(
                 &BoostNumber::precision),
             bp::args("self", "digits10"), "Set the precision of this.")

        .def("__float__", &cast<double>, bp::arg("self"), "Cast to float.")
        .def("__int__", &cast<int64_t>, bp::arg("self"), "Cast to int.")

        .def("__str__", &print, bp::arg("self"))
        .def("__repr__", &print, bp::arg("self"))

        .def("set_display_precision", &set_display_precision, bp::arg("digit"),
             "Set the precision when printing values.")
        .staticmethod("set_display_precision")

        .def("get_display_precision", &get_display_precision,
             "Get the precision when printing values.",
             bp::return_value_policy<bp::copy_non_const_reference>())
        .staticmethod("get_display_precision")

        // #ifndef PINOCCHIO_PYTHON_NO_SERIALIZATION
        //         .def_pickle(Pickle())
        // #endif
        ;
  }

  static void expose(const std::string &type_name) {
    bp::class_<BoostNumber>(type_name.c_str(), "", bp::no_init)
        .def(BoostNumberPythonVisitor<BoostNumber>());

    eigenpy::registerNewType<BoostNumber>();
    eigenpy::registerCommonUfunc<BoostNumber>();

#define IMPLICITLY_CONVERTIBLE(T1, T2) bp::implicitly_convertible<T1, T2>();
    //  bp::implicitly_convertible<T2,T1>();

    IMPLICITLY_CONVERTIBLE(double, BoostNumber);
    IMPLICITLY_CONVERTIBLE(float, BoostNumber);
    IMPLICITLY_CONVERTIBLE(long int, BoostNumber);
    IMPLICITLY_CONVERTIBLE(int, BoostNumber);
    IMPLICITLY_CONVERTIBLE(long, BoostNumber);
    IMPLICITLY_CONVERTIBLE(unsigned int, BoostNumber);
    IMPLICITLY_CONVERTIBLE(unsigned long int, BoostNumber);
    IMPLICITLY_CONVERTIBLE(bool, BoostNumber);

#undef IMPLICITLY_CONVERTIBLE

    eigenpy::registerCast<BoostNumber, double>(false);
    eigenpy::registerCast<double, BoostNumber>(true);
    eigenpy::registerCast<BoostNumber, float>(false);
    eigenpy::registerCast<float, BoostNumber>(true);
    eigenpy::registerCast<BoostNumber, long>(false);
    eigenpy::registerCast<long, BoostNumber>(true);
    eigenpy::registerCast<BoostNumber, int>(false);
    eigenpy::registerCast<int, BoostNumber>(true);
    ;
    eigenpy::registerCast<BoostNumber, int64_t>(false);
    eigenpy::registerCast<int64_t, BoostNumber>(true);
  }

 private:
  template <typename T>
  static T cast(const BoostNumber &self) {
    return static_cast<T>(self);
  }

  static std::string print(const BoostNumber &self) {
    return self.str(get_display_precision(), std::ios_base::dec);
  }

  static void set_display_precision(const int digit) {
    get_display_precision() = digit;
  }

  static int &get_display_precision() {
    static int precision = BoostNumber::default_precision();
    return precision;
  }
};

// this derived directly from the code at
// https://github.com/stack-of-tasks/eigenpy/issues/365, in which this example
// was requested

template <typename BoostNumber>
struct BoostComplexPythonVisitor
    : public boost::python::def_visitor<
          BoostComplexPythonVisitor<BoostNumber> > {
 public:
  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>("Default constructor.", bp::arg("self")))
        .def(bp::init<BoostNumber>("Copy constructor.",
                                   bp::args("self", "value")))
        .def(bp::init<std::string>("Constructor from a string.",
                                   bp::args("self", "str_value")))
        .def(bp::init<std::string, std::string>(
            "Constructor from a pair of strings.",
            bp::args("self", "real", "imag")))

        .def(bp::init<int, int>("Constructor from a pair of integers.",
                                bp::args("self", "real", "imag")))

        .def(bp::self + bp::self)
        .def(bp::self += bp::self)
        .def(bp::self - bp::self)
        .def(bp::self -= bp::self)
        .def(bp::self * bp::self)
        .def(bp::self *= bp::self)
        .def(bp::self / bp::self)
        .def(bp::self /= bp::self)

        .def(bp::self == bp::self)
        .def(bp::self != bp::self)
        .def(bp::self_ns::pow(bp::self_ns::self, long()))

        .def("str", &BoostNumber::str,
             bp::args("self", "precision", "scientific"))

        .def("default_precision",
             static_cast<unsigned (*)()>(BoostNumber::default_precision),
             "Get the default precision of the class.")
        .def("default_precision",
             static_cast<void (*)(unsigned)>(BoostNumber::default_precision),
             bp::arg("digits10"), "Set the default precision of the class.")
        .staticmethod("default_precision")

        .add_property("real", &get_real, &set_real)
        .add_property("imag", &get_imag, &set_imag)

        .def("precision",
             static_cast<unsigned (BoostNumber::*)() const>(
                 &BoostNumber::precision),
             bp::arg("self"), "Get the precision of this.")
        .def("precision",
             static_cast<void (BoostNumber::*)(unsigned)>(
                 &BoostNumber::precision),
             bp::args("self", "digits10"), "Set the precision of this.")

        .def("__str__", &print, bp::arg("self"))
        .def("__repr__", &print, bp::arg("self"))

        .def("set_display_precision", &set_display_precision, bp::arg("digit"),
             "Set the precision when printing values.")
        .staticmethod("set_display_precision")

        .def("get_display_precision", &get_display_precision,
             "Get the precision when printing values.",
             bp::return_value_policy<bp::copy_non_const_reference>())
        .staticmethod("get_display_precision")

        ;
  }

  static void set_real(BoostNumber &c, mpfr_float const &r) { c.real(r); }
  static mpfr_float get_real(BoostNumber const &c) { return c.real(); }

  static void set_imag(BoostNumber &c, mpfr_float const &r) { c.imag(r); }
  static mpfr_float get_imag(BoostNumber const &c) { return c.imag(); }

  static std::string print(const BoostNumber &self) {
    return self.str(get_display_precision(), std::ios_base::dec);
  }

  static void set_display_precision(const int digit) {
    get_display_precision() = digit;
  }

  static int &get_display_precision() {
    static int precision = BoostNumber::default_precision();
    return precision;
  }
};

#endif
