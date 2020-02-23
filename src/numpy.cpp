/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/numpy.hpp"

namespace eigenpy
{
  void import_numpy()
  {
    if(_import_array() < 0)
    {
      PyErr_Print();
      PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    }
    //        std::cout << "init _import_array " << std::endl;
  }
}
