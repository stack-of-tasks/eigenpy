/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_exception_hpp__
#define __eigenpy_exception_hpp__

#include <exception>
#include <string>

#include "eigenpy/fwd.hpp"

namespace eigenpy {
/*
 * Eigenpy exception. They can be catch with python (equivalent
 * eigenpy.exception class).
 */
class Exception : public std::exception {
 public:
  Exception() : message() {}
  Exception(const std::string &msg) : message(msg) {}
  const char *what() const throw() { return this->getMessage().c_str(); }
  ~Exception() throw() {}
  virtual const std::string &getMessage() const { return message; }
  std::string copyMessage() const { return getMessage(); }

  /* Call this static function to "enable" the translation of this C++ exception
   * in Python. */
  static void registerException();

 private:
  static void translateException(Exception const &e);
  static PyObject *pyType;

 protected:
  std::string message;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_exception_hpp__
