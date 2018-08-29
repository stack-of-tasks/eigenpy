/*
 * Copyright 2014,2018 Nicolas Mansard and Justin Carpentier, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <exception>
#include <string>

#ifndef __eigenpy_exception_hpp__
#define __eigenpy_exception_hpp__

namespace eigenpy
{
  /*
   * Eigenpy exception. They can be catch with python (equivalent eigenpy.exception class).
   */
  class Exception : public std::exception
  {
  public:
    Exception() : message() {}
    Exception(const std::string & msg) : message(msg) {}
    const char *what() const throw()
    {
      return this->getMessage().c_str();
    }
    ~Exception() throw() {}
    virtual const std::string & getMessage() const { return message; }
    std::string copyMessage() const { return getMessage(); }

    /* Call this static function to "enable" the translation of this C++ exception in Python. */
    static void registerException();

  private:
    static void translateException( Exception const & e );
    static PyObject * pyType;
  protected:
    std::string message;
   };

} // namespace eigenpy

#endif // ifndef __eigenpy_exception_hpp__
