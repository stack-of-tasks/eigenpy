//
// Copyright (c) 2020-2021 INRIA
// code aptapted from
// https://github.com/numpy/numpy/blob/41977b24ae011a51f64faa75cb524c7350fdedd9/numpy/core/src/umath/_rational_tests.c.src
//

#ifndef __eigenpy_ufunc_hpp__
#define __eigenpy_ufunc_hpp__

#include "eigenpy/register.hpp"
#include "eigenpy/user-type.hpp"

namespace eigenpy {
namespace internal {

#ifdef NPY_1_19_API_VERSION
#define EIGENPY_NPY_CONST_UFUNC_ARG const
#else
#define EIGENPY_NPY_CONST_UFUNC_ARG
#endif

template <typename T>
void matrix_multiply(char **args, npy_intp const *dimensions,
                     npy_intp const *steps) {
  /* pointers to data for input and output arrays */
  char *ip1 = args[0];
  char *ip2 = args[1];
  char *op = args[2];

  /* lengths of core dimensions */
  npy_intp dm = dimensions[0];
  npy_intp dn = dimensions[1];
  npy_intp dp = dimensions[2];

  /* striding over core dimensions */
  npy_intp is1_m = steps[0];
  npy_intp is1_n = steps[1];
  npy_intp is2_n = steps[2];
  npy_intp is2_p = steps[3];
  npy_intp os_m = steps[4];
  npy_intp os_p = steps[5];

  /* core dimensions counters */
  npy_intp m, p;

  /* calculate dot product for each row/column vector pair */
  for (m = 0; m < dm; m++) {
    for (p = 0; p < dp; p++) {
      SpecialMethods<T>::dotfunc(ip1, is1_n, ip2, is2_n, op, dn, NULL);

      /* advance to next column of 2nd input array and output array */
      ip2 += is2_p;
      op += os_p;
    }

    /* reset to first column of 2nd input array and output array */
    ip2 -= is2_p * p;
    op -= os_p * p;

    /* advance to next row of 1st input array and output array */
    ip1 += is1_m;
    op += os_m;
  }
}

template <typename T>
void gufunc_matrix_multiply(char **args,
                            npy_intp EIGENPY_NPY_CONST_UFUNC_ARG *dimensions,
                            npy_intp EIGENPY_NPY_CONST_UFUNC_ARG *steps,
                            void *NPY_UNUSED(func)) {
  /* outer dimensions counter */
  npy_intp N_;

  /* length of flattened outer dimensions */
  npy_intp dN = dimensions[0];

  /* striding over flattened outer dimensions for input and output arrays */
  npy_intp s0 = steps[0];
  npy_intp s1 = steps[1];
  npy_intp s2 = steps[2];

  /*
   * loop through outer dimensions, performing matrix multiply on
   * core dimensions for each loop
   */
  for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
    matrix_multiply<T>(args, dimensions + 1, steps + 3);
  }
}

#define EIGENPY_REGISTER_BINARY_OPERATOR(name, op)                           \
  template <typename T1, typename T2, typename R>                            \
  void binary_op_##name(                                                     \
      char **args, EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *dimensions,         \
      EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *steps, void * /*data*/) {        \
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions; \
    char *i0 = args[0], *i1 = args[1], *o = args[2];                         \
    int k;                                                                   \
    for (k = 0; k < n; k++) {                                                \
      T1 &x = *static_cast<T1 *>(static_cast<void *>(i0));                   \
      T2 &y = *static_cast<T2 *>(static_cast<void *>(i1));                   \
      R &res = *static_cast<R *>(static_cast<void *>(o));                    \
      res = x op y;                                                          \
      i0 += is0;                                                             \
      i1 += is1;                                                             \
      o += os;                                                               \
    }                                                                        \
  }                                                                          \
                                                                             \
  template <typename T>                                                      \
  void binary_op_##name(                                                     \
      char **args, EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *dimensions,         \
      EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *steps, void *data) {             \
    binary_op_##name<T, T, T>(args, dimensions, steps, data);                \
  }

EIGENPY_REGISTER_BINARY_OPERATOR(add, +)
EIGENPY_REGISTER_BINARY_OPERATOR(subtract, -)
EIGENPY_REGISTER_BINARY_OPERATOR(multiply, *)
EIGENPY_REGISTER_BINARY_OPERATOR(divide, /)
EIGENPY_REGISTER_BINARY_OPERATOR(equal, ==)
EIGENPY_REGISTER_BINARY_OPERATOR(not_equal, !=)
EIGENPY_REGISTER_BINARY_OPERATOR(less, <)
EIGENPY_REGISTER_BINARY_OPERATOR(greater, >)
EIGENPY_REGISTER_BINARY_OPERATOR(less_equal, <=)
EIGENPY_REGISTER_BINARY_OPERATOR(greater_equal, >=)

#define EIGENPY_REGISTER_UNARY_OPERATOR(name, op)                     \
  template <typename T, typename R>                                   \
  void unary_op_##name(                                               \
      char **args, EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *dimensions,  \
      EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *steps, void * /*data*/) { \
    npy_intp is = steps[0], os = steps[1], n = *dimensions;           \
    char *i = args[0], *o = args[1];                                  \
    int k;                                                            \
    for (k = 0; k < n; k++) {                                         \
      T &x = *static_cast<T *>(static_cast<void *>(i));               \
      R &res = *static_cast<R *>(static_cast<void *>(o));             \
      res = op x;                                                     \
      i += is;                                                        \
      o += os;                                                        \
    }                                                                 \
  }                                                                   \
                                                                      \
  template <typename T>                                               \
  void unary_op_##name(                                               \
      char **args, EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *dimensions,  \
      EIGENPY_NPY_CONST_UFUNC_ARG npy_intp *steps, void *data) {      \
    unary_op_##name<T, T>(args, dimensions, steps, data);             \
  }

EIGENPY_REGISTER_UNARY_OPERATOR(negative, -)

}  // namespace internal

#define EIGENPY_REGISTER_BINARY_UFUNC(name, code, T1, T2, R)                   \
  {                                                                            \
    PyUFuncObject *ufunc =                                                     \
        (PyUFuncObject *)PyObject_GetAttrString(numpy, #name);                 \
    int _types[3] = {Register::getTypeCode<T1>(), Register::getTypeCode<T2>(), \
                     Register::getTypeCode<R>()};                              \
    if (!ufunc) {                                                              \
      /*goto fail; \*/                                                         \
    }                                                                          \
    if (sizeof(_types) / sizeof(int) != ufunc->nargs) {                        \
      PyErr_Format(PyExc_AssertionError,                                       \
                   "ufunc %s takes %d arguments, our loop takes %lu", #name,   \
                   ufunc->nargs,                                               \
                   (unsigned long)(sizeof(_types) / sizeof(int)));             \
      Py_DECREF(ufunc);                                                        \
    }                                                                          \
    if (PyUFunc_RegisterLoopForType((PyUFuncObject *)ufunc, code,              \
                                    internal::binary_op_##name<T1, T2, R>,     \
                                    _types, 0) < 0) {                          \
      /*Py_DECREF(ufunc);*/                                                    \
      /*goto fail; \*/                                                         \
    }                                                                          \
    Py_DECREF(ufunc);                                                          \
  }

#define EIGENPY_REGISTER_UNARY_UFUNC(name, code, T, R)                        \
  {                                                                           \
    PyUFuncObject *ufunc =                                                    \
        (PyUFuncObject *)PyObject_GetAttrString(numpy, #name);                \
    int _types[2] = {Register::getTypeCode<T>(), Register::getTypeCode<R>()}; \
    if (!ufunc) {                                                             \
      /*goto fail; \*/                                                        \
    }                                                                         \
    if (sizeof(_types) / sizeof(int) != ufunc->nargs) {                       \
      PyErr_Format(PyExc_AssertionError,                                      \
                   "ufunc %s takes %d arguments, our loop takes %lu", #name,  \
                   ufunc->nargs,                                              \
                   (unsigned long)(sizeof(_types) / sizeof(int)));            \
      Py_DECREF(ufunc);                                                       \
    }                                                                         \
    if (PyUFunc_RegisterLoopForType((PyUFuncObject *)ufunc, code,             \
                                    internal::unary_op_##name<T, R>, _types,  \
                                    0) < 0) {                                 \
      /*Py_DECREF(ufunc);*/                                                   \
      /*goto fail; \*/                                                        \
    }                                                                         \
    Py_DECREF(ufunc);                                                         \
  }

template <typename Scalar>
void registerCommonUfunc() {
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
  EIGENPY_REGISTER_BINARY_UFUNC(greater, type_code, Scalar, Scalar, bool);
  EIGENPY_REGISTER_BINARY_UFUNC(less, type_code, Scalar, Scalar, bool);
  EIGENPY_REGISTER_BINARY_UFUNC(greater_equal, type_code, Scalar, Scalar, bool);
  EIGENPY_REGISTER_BINARY_UFUNC(less_equal, type_code, Scalar, Scalar, bool);

  // Unary operators
  EIGENPY_REGISTER_UNARY_UFUNC(negative, type_code, Scalar, Scalar);

  Py_DECREF(numpy);
}

}  // namespace eigenpy

#endif  // __eigenpy_ufunc_hpp__
