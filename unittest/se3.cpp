#include "libse3.hpp"

/* ------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------------------- */
  
#include "eigenpy/simple.hpp"
#include "eigenpy/geometry.hpp"
#include <boost/python.hpp>
namespace bp = boost::python;

BOOST_PYTHON_MODULE(se3)
{
  eigenpy::enableEigenPy();
  eigenpy::exposeAngleAxis();
  eigenpy::exposeQuaternion();
  
  buildFsMap();
  
  for( std::map<std::string,functiondoc_t>::iterator it=fs.begin();it!=fs.end();++it )
    {
      bp::def(it->first.c_str(),it->second.first,it->second.second.c_str());
    }

  // bp::def("geom1",&se3::geom<se3::RigidMotion::Rp>,"Geometry with simple R+p");
  // bp::def("geom2",&se3::geom<se3::RigidMotion::EigenM>,"Geometry with Eigen::Affine3d");
  // bp::def("geomRx",&se3::geom<se3::RigidMotion::Rx>,"Geometry with Revolute x");
  // bp::def("geomLowBound",&se3::geom<se3::RigidMotion::Simple33>,"Theoretial lower bound for geometry");

  // bp::def("kine1",&se3::kine<se3::RigidMotion::Rp,    se3::Motion6::R32>,"Kine with SE3::R+p and M6::R32");
  // bp::def("kine2",&se3::kine<se3::RigidMotion::EigenM,se3::Motion6::R32>,"Kine with SE3::Eigen and M6::R32");
  // bp::def("kine3",&se3::kine<se3::RigidMotion::Rp,    se3::Motion6::VW>,"Kine with SE3::R+p and M6::VW");
  // bp::def("kine4",&se3::kine<se3::RigidMotion::EigenM,se3::Motion6::VW>,"Kine with SE3::Eigen and M6::VW");
  // bp::def("kineRx",&se3::kine<se3::RigidMotion::Rx,    se3::Motion6::R32>,"Kine with SE3::Rx and M6::R32");
  // bp::def("kineLowBound",&se3::kine<se3::RigidMotion::Simple33,se3::Motion6::Simple32>,"Theoretial lower bound for kine");

  // bp::def("kinegeom1",&se3::kinegeom<se3::RigidMotion::Rp,    se3::Motion6::R32>,"Kine-Geom with SE3::R+p and M6::R32");
  // bp::def("kinegeom2",&se3::kinegeom<se3::RigidMotion::EigenM,se3::Motion6::R32>,"Kine-Geom with SE3::Eigen and M6::R32");
  // bp::def("kinegeom3",&se3::kinegeom<se3::RigidMotion::Rp,    se3::Motion6::VW>,"Kine-Geom with SE3::R+p and M6::VW");
  // bp::def("kinegeom4",&se3::kinegeom<se3::RigidMotion::EigenM,se3::Motion6::VW>,"Kine-Geom with SE3::Eigen and M6::VW");
  // bp::def("kinegeomRx",&se3::kinegeom<se3::RigidMotion::Rx,    se3::Motion6::R32>,"Kine-Geom with SE3::Rx and M6::R32");
  // bp::def("kinegeomLowBound",&se3::kinegeom<se3::RigidMotion::Simple33,se3::Motion6::Simple32>,"Theoretial lower bound for kine-geom");

  // bp::def("tcos",&testCos);

}
 
