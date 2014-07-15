#include <Eigen/Core>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

namespace eigenpy
{
  template< typename MatType, int IsVector>
  struct MapNumpyTraits {};
 
  template< typename MatType >
  struct MapNumpy
  {
    typedef MapNumpyTraits<MatType, MatType::IsVectorAtCompileTime> Impl;
    typedef typename Impl::EigenMap EigenMap;

    static inline EigenMap map( PyArrayObject* pyArray );
   };

  class exception : public std::exception
  {
  public:
    exception(std::string msg) : message(msg) {}
    const char *what() const throw()
    {
      return this->message.c_str();
    }
    ~exception() throw() {}
    std::string getMessage() { return message; }
    static void registerException();

  private:
    static void translateException( exception const & e );
    static PyObject * pyType;
    std::string message;
   };

  template<typename MatType>
  void enableEigenPySpecific();

  void enableEigenPy()
  {
    exception::registerException();

    enableEigenPySpecific<Eigen::MatrixXd>();
    enableEigenPySpecific<Eigen::Matrix2d>();
    enableEigenPySpecific<Eigen::Matrix3d>();
    enableEigenPySpecific<Eigen::Matrix4d>();

    enableEigenPySpecific<Eigen::VectorXd>();
    enableEigenPySpecific<Eigen::Vector2d>();
    enableEigenPySpecific<Eigen::Vector3d>();
    enableEigenPySpecific<Eigen::Vector4d>();
  }
}

/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */

namespace eigenpy
{
  template <typename SCALAR>  struct NumpyEquivalentType {};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};

  /* --- MAP ON NUMPY ----------------------------------------------------------- */
  template<typename MatType>
  struct MapNumpyTraits<MatType,0>
  {
    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;
    typedef Eigen::Map<MatType,0,Stride> EigenMap;
    typedef typename MatType::Scalar T;

    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) == 2 );
      
      const int R = PyArray_DIMS(pyArray)[0];
      const int C = PyArray_DIMS(pyArray)[1];
      const int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride1 = PyArray_STRIDE(pyArray, 0) / itemsize;
      const int stride2 = PyArray_STRIDE(pyArray, 1) / itemsize;
      
      if( (MatType::RowsAtCompileTime!=R)
	  && (MatType::RowsAtCompileTime!=Eigen::Dynamic) )
	{ throw eigenpy::exception("The number of rows does not fit with the matrix type."); }
      if( (MatType::ColsAtCompileTime!=C)
	  && (MatType::ColsAtCompileTime!=Eigen::Dynamic) )
	{  throw eigenpy::exception("The number of columns does not fit with the matrix type."); }

      T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));
      return EigenMap( pyData, R,C, Stride(stride2,stride1) );
    }
  };

  template<typename MatType>
  struct MapNumpyTraits<MatType,1>
  {
    typedef Eigen::InnerStride<Eigen::Dynamic> Stride;
    typedef Eigen::Map<MatType,0,Stride> EigenMap;
    typedef typename MatType::Scalar T;
 
    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) <= 2 );

      int rowMajor;
      if(  PyArray_NDIM(pyArray)==1 ) rowMajor = 0;
      else rowMajor = (PyArray_DIMS(pyArray)[0]>PyArray_DIMS(pyArray)[1])?0:1;

      const int R = PyArray_DIMS(pyArray)[rowMajor];
      const int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride = PyArray_STRIDE(pyArray, rowMajor) / itemsize;;

      if( (MatType::MaxSizeAtCompileTime==R)
	      || (MatType::MaxSizeAtCompileTime==Eigen::Dynamic) )
	{ throw eigenpy::exception("The number of elements does not fit with the vector type."); }

      T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));
      return EigenMap( pyData, R, 1, Stride(stride) );
    }
  };

  template< typename MatType >
  typename MapNumpy<MatType>::EigenMap MapNumpy<MatType>::map( PyArrayObject* pyArray )
  {
    return Impl::mapImpl(pyArray); 
  }

  /* --- TO PYTHON -------------------------------------------------------------- */
  template< typename MatType >
  struct EigenToPy
  {
    static PyObject* convert(MatType const& mat)
    {
      typedef typename MatType::Scalar T;
      const int R  = mat.rows(), C = mat.cols();

      npy_intp shape[2] = { R,C };
      PyArrayObject* pyArray = (PyArrayObject*)
	PyArray_SimpleNew(2, shape, NumpyEquivalentType<T>::type_code);

      MapNumpy<MatType>::map(pyArray) = mat;

      return (PyObject*)pyArray;
    }
  };
  
  /* --- FROM PYTHON ------------------------------------------------------------ */
  namespace bp = boost::python;

  template<typename MatType>
  struct EigenFromPy
  {
    EigenFromPy()
    {
      bp::converter::registry::push_back
	(&convertible,&construct,bp::type_id<MatType>());
    }
 
    // Determine if obj_ptr can be converted in a Eigenvec
    static void* convertible(PyObject* obj_ptr)
    {
      typedef typename MatType::Scalar T;

      if (!PyArray_Check(obj_ptr)) 
	{
	  std::cerr << "The python object is not a numpy array." << std::endl;
	  return 0;
	}

      if (PyArray_NDIM(obj_ptr) != 2)
	if ( (PyArray_NDIM(obj_ptr) !=1) || (! MatType::IsVectorAtCompileTime) )
	  {
	    std::cerr << "The number of dimension of the object is not correct." << std::endl;
	    return 0;
	  }

      if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<T>::type_code)
	{
	  std::cerr << "The internal type as no Eigen equivalent." << std::endl;
	  return 0;
	}

      if (!(PyArray_FLAGS(obj_ptr) & NPY_ALIGNED))
	{
	  std::cerr << "NPY non-aligned matrices are not implemented." << std::endl;
	  return 0;
	}
      
      return obj_ptr;
    }
 
    // Convert obj_ptr into a Eigenvec
    static void construct(PyObject* pyObj,
			  bp::converter::rvalue_from_python_stage1_data* memory)
    {
      typedef typename MatType::Scalar T;
      using namespace Eigen;

      PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
      typename MapNumpy<MatType>::EigenMap numpyMap = MapNumpy<MatType>::map(pyArray);

      void* storage = ((bp::converter::rvalue_from_python_storage<MatType>*)
		       (memory))->storage.bytes;
      MatType & eigenMatrix = * new(storage) MatType(numpyMap.rows(),numpyMap.cols());
      memory->convertible = storage;

      eigenMatrix = numpyMap;
    }
  };

  template<typename MatType>
  void enableEigenPySpecific()
  {
    import_array();
    boost::python::to_python_converter<MatType,
				       eigenpy::EigenToPy<MatType> >();
    eigenpy::EigenFromPy<MatType>();
  }

  /* --- EXCEPTION ----------------------------------------------------------------- */
  PyObject * exception::pyType;

  void exception::translateException( exception const & e )
  {
    assert(NULL!=pyType);
    // Return an exception object of type pyType and value object(e).
    PyErr_SetObject(pyType,boost::python::object(e).ptr());
  }

  void exception::registerException()
  {
    pyType = boost::python::class_<eigenpy::exception>
      ("exception",boost::python::init<std::string>())
      .add_property("message", &eigenpy::exception::getMessage)
      .ptr();

    boost::python::register_exception_translator<eigenpy::exception>
      (&eigenpy::exception::translateException);
  }

} // namespace eigenpy
