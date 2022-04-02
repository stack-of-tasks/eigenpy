/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/exception.hpp"

#include <boost/python/exception_translator.hpp>

#include "eigenpy/registration.hpp"

namespace eigenpy {
PyObject* Exception::pyType;

void Exception::translateException(Exception const& e) {
  assert(NULL != pyType);
  // Return an exception object of type pyType and value object(e).
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

void Exception::registerException() {
  if (check_registration<eigenpy::Exception>()) return;

  pyType = boost::python::class_<eigenpy::Exception>(
               "Exception", boost::python::init<std::string>())
               .add_property("message", &eigenpy::Exception::copyMessage)
               .ptr();

  boost::python::register_exception_translator<eigenpy::Exception>(
      &eigenpy::Exception::translateException);
}

}  // namespace eigenpy
