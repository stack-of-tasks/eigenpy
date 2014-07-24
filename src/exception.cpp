#include <eigenpy/exception.hpp>


namespace eigenpy
{
  PyObject * Exception::pyType;

  void Exception::translateException( Exception const & e )
  {
    assert(NULL!=pyType);
    // Return an exception object of type pyType and value object(e).
    PyErr_SetObject(Exception::pyType,boost::python::object(e).ptr());
  }

  void Exception::registerException()
  {
    pyType = boost::python::class_<eigenpy::Exception>
      ("Exception",boost::python::init<std::string>())
      .add_property("message", &eigenpy::Exception::copyMessage)
      .ptr();

    boost::python::register_exception_translator<eigenpy::Exception>
      (&eigenpy::Exception::translateException);
  }

} // namespace eigenpy
