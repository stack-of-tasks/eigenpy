//
// Copyright (c) 2024 INRIA
//
//

#ifndef __eigenpy_utils_python_compat_hpp__
#define __eigenpy_utils_python_compat_hpp__

#if PY_MAJOR_VERSION >= 3

#define PyInt_Check PyLong_Check

#define PyStr_Check PyUnicode_Check
#define PyStr_FromString PyUnicode_FromString

#else

#define PyStr_Check PyString_Check
#define PyStr_FromString PyString_FromString

#endif

#endif  // ifndef __eigenpy_utils_python_compat_hpp__
