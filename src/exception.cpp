/*
 * Copyright (c) 2015-2018 LAAS-CNRS
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

#include "eigenpy/exception.hpp"
#include "eigenpy/registration.hpp"

#include <boost/python/exception_translator.hpp>


namespace eigenpy
{
  PyObject * Exception::pyType;

  void Exception::translateException( Exception const & e )
  {
    assert(NULL!=pyType);
    // Return an exception object of type pyType and value object(e).
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }

  void Exception::registerException()
  {
    if(check_registration<eigenpy::Exception>()) return;
    
    pyType = boost::python::class_<eigenpy::Exception>
      ("Exception",boost::python::init<std::string>())
      .add_property("message", &eigenpy::Exception::copyMessage)
      .ptr();

    boost::python::register_exception_translator<eigenpy::Exception>
      (&eigenpy::Exception::translateException);
  }

} // namespace eigenpy
