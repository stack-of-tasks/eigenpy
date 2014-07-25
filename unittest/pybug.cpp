
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sys/time.h>
#include<Eigen/StdVector>
#include <boost/python.hpp>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4d)


Eigen::Matrix4d f0()      { return Eigen::Matrix4d(); }

double faffine()
{
  Eigen::Affine3d oX1,oX2,oX3;

  struct timeval t0,t1;
  gettimeofday(&t0,NULL);
  for( int i=1; i<10*1000; ++i )
      {
	oX1 = oX2*oX3;
	oX1 = oX2*oX3;
	oX1 = oX2*oX3;
      }
  gettimeofday(&t1,NULL);
  return ((t1.tv_sec - t0.tv_sec)+1e-6*(t1.tv_usec - t0.tv_usec))*1000;
}


double fmatrix()
{
  std::vector< Eigen::Matrix4d > sXp(4);
  Eigen::Matrix4d oX1;
  
  for( int i=1; i<100; ++i )
    {
      oX1 = f0();
      oX1 = f0();
      oX1 = f0();
      oX1 = f0();
    }
  return 0.0;
}


BOOST_PYTHON_MODULE(pybug)
{
  boost::python::def("kinegeom1",&fmatrix);
  boost::python::def("kinegeom2",&faffine);
}
 
/* 
from se3 import kinegeom1,kinegeom2

for i in range(10):
    print kinegeom2()

print " *************"
print kinegeom1()
print " *************"

for i in range(10):
    print kinegeom2()


*/

