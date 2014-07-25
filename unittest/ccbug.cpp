// g++  ../unittest/ccbug.cpp -I /usr/include/eigen3/ -o ccbug -O3 -DNDEBUG && ./ccbug 

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sys/time.h>
#include <iostream>

int main()
{
  Eigen::Affine3d oX1,oX2,oX3;
  struct timeval t0,t1;

  for(int r=0;r<10;++r)
    {
      gettimeofday(&t0,NULL);
      for( int i=1; i<10*1000; ++i )
	{
	  oX1 = oX2*oX3; // Without one of this 3 lines, the computation cost is around 1e-5 us
	  oX1 = oX2*oX3; // With the tree lines, it is 75 us.
	  oX1 = oX2*oX3;
	}
      gettimeofday(&t1,NULL);
      std::cout <<  ((t1.tv_sec - t0.tv_sec)+1e-6*(t1.tv_usec - t0.tv_usec))*1000 << std::endl;
    }

  return 0;
}
