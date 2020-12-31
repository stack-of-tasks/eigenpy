//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_swig_hpp__
#define __eigenpy_swig_hpp__

namespace eigenpy
{
  struct PySwigObject
  {
    PyObject_HEAD
    void * ptr;
    const char * desc;
  };
}

#endif // ifndef __eigenpy_swig_hpp__
