#ifndef __eigenpy_details_rvalue_from_python_data_hpp__
#define __eigenpy_details_rvalue_from_python_data_hpp__

#include <boost/python/converter/rvalue_from_python_data.hpp>
#include <Eigen/Core>

namespace boost
{
  namespace python
  {
    namespace converter
    {
  
      /// \brief Template specialization of rvalue_from_python_data
      template<typename Derived>
      struct rvalue_from_python_data<Eigen::MatrixBase<Derived> const & >
      : rvalue_from_python_storage<Eigen::MatrixBase<Derived> const & >
      {
        typedef Eigen::MatrixBase<Derived> const & T;
        
# if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) \
&& (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) \
&& (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) \
&& !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing this */
        // This must always be a POD struct with m_data its first member.
        BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<T>,stage1) == 0);
# endif
        
        // The usual constructor
        rvalue_from_python_data(rvalue_from_python_stage1_data const & _stage1)
        {
          this->stage1 = _stage1;
        }
        
        // This constructor just sets m_convertible -- used by
        // implicitly_convertible<> to perform the final step of the
        // conversion, where the construct() function is already known.
        rvalue_from_python_data(void* convertible)
        {
          this->stage1.convertible = convertible;
        }
        
        // Destroys any object constructed in the storage.
        ~rvalue_from_python_data()
        {
          if (this->stage1.convertible == this->storage.bytes)
            static_cast<Derived *>((void *)this->storage.bytes)->~Derived();
        }
      };
      
    }
  }
} // namespace boost::python::converter

#endif // ifndef __eigenpy_details_rvalue_from_python_data_hpp__
