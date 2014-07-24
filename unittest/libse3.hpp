//#define EIGEN_DONT_VECTORIZE 
//#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
// #include "eigenpy/simple.hpp"
// #include "eigenpy/geometry.hpp"
#include "timer.h"

// #include <boost/python.hpp>
// namespace bp = boost::python;


namespace se3
{
  typedef Eigen::Matrix3d Matrix3;
  typedef Eigen::Vector3d Vector3;
  
  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  
  namespace Motion6
  {
    template<typename M>
    M random();
    
    class R32 : public Eigen::Matrix<double,3,2>
    {
      enum { LINEAR = 0, ANGULAR = 1 };

    public:
      typedef Eigen::Matrix<double,3,2> DenseBase;
      DenseBase & base() { return *this; }
      const DenseBase & base() const { return *this; }
  
      Eigen::Block<DenseBase,3,1> linear()                    { return base().block<3,1>(0,LINEAR); }
      const Eigen::Block<const DenseBase,3,1> linear()  const { return base().block<3,1>(0,LINEAR); }

      Eigen::Block<DenseBase,3,1> angular()                    { return base().block<3,1>(0,ANGULAR); }
      const Eigen::Block<const DenseBase,3,1>  angular() const { return base().block<3,1>(0,ANGULAR); }
  
      R32() : DenseBase( DenseBase::Zero()) {}
      R32(const DenseBase & d) : DenseBase(d) {}
      R32& operator= ( const DenseBase & b) { base()=b; return *this; }

      template<typename Dl,typename Da>
      R32( const MatrixBase<Dl> & l, const MatrixBase<Da> & a )
      {
	EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Dl,3);
	EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Da,3);
	linear() = l; angular() = a; 
      }
    };

    template<>
    R32 random<R32>() 
    {
      R32 res;
      res.base() = R32::DenseBase::Random(); 
      return res;
    }
    
    struct VW
    {
      Vector3 v,w;
      template<typename D1,typename D2>
      VW (const Eigen::MatrixBase<D1> & _v, const Eigen::MatrixBase<D2> & _w )
	: v(_v),w(_w) 
      {
	EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(D1,3);
	EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(D2,3);
      }
      VW () : v(),w() {} 
    };

    template<>
    VW random<VW>()
    {
      VW res;
      res.v = Vector3::Random(); res.w = Vector3::Random();
      return res;
    }

    typedef Eigen::Matrix<double,3,2> Simple32;
    template<> Simple32 random<Simple32>()    { return Simple32::Random(); }

  } // namespace Motion6

  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  
  namespace RigidMotion
  {
    template<typename M>
    M random();

    struct Rp
    {
      Matrix3 R; Vector3 p;
      Rp(const Matrix3 & _R,const Vector3 & _p) : R(_R),p(_p) {}
      Rp() : R(Matrix3::Identity()),p() {}
      friend Rp operator* (const Rp & m1, const Rp & m2 )
      { return Rp(m1.R*m2.R,m1.p+m1.R*m2.p); }
    };

    template<>
    Rp random<Rp>()
    {
      Rp res;
      Eigen::Quaterniond q(Eigen::Vector4d::Random());
      q.normalize(); res.R = q.matrix();
      res.p = Eigen::Vector3d::Random();
      return res;
    }

    typedef Eigen::Affine3d EigenM;

    template<>
    EigenM random<EigenM>() 
    {
      Eigen::Quaterniond q(Eigen::Vector4d::Random());
      q.normalize();
      Vector3 p = Eigen::Vector3d::Random();
      return Eigen::Translation3d(p)*q;
    }

    struct Rx
    {
      Vector3 p;
      double ca,sa,a;
      template<typename D>
      Rx(double a_,const Eigen::MatrixBase<D>&p_): p(p_),ca(cos(a_)),sa((a_)),a(a_) {}
      template<typename D>
      Rx(double a_,double ca_,double sa_,const Eigen::MatrixBase<D>&p_): p(p_),ca(ca_),sa(sa_),a(a_) {}
      Rx() : p(),ca(1),sa(0),a(0) {}
      friend Rx operator* (const Rx& m1,const Rx& m2)
      {
	//Rx res(m1.a+m2.a,m1.p);
	// res.p[1] +=  m2.p[1]*m2.ca + m2.p[2]*m2.sa;
	// res.p[2] += -m2.p[1]*m2.sa + m2.p[2]*m2.ca;
	//return res;
	//c12 = c1*c2 - s1*s2
	//s12 = c1*s1 + c2*c2
	Rx res(m1.a+m2.a, m1.ca*m2.ca-m1.sa*m2.ca, m1.ca*m2.sa+m2.ca*m1.sa,   m1.p);
	res.p[1] +=  m2.p[1]*m2.ca + m2.p[2]*m2.sa;
	res.p[2] += -m2.p[1]*m2.sa + m2.p[2]*m2.ca;
	return res;
      }
    };

    template<>
    Rx random<Rx>()
    {
      Eigen::Matrix<double,1,1> a = Eigen::Matrix<double,1,1>::Random();
      Vector3 p = Vector3::Random();
      return Rx(a[0],p);
    }

    typedef Eigen::Matrix3d Simple33;
    template<> Simple33 random<Simple33>() { return Simple33::Random(); }

  } // namespace RigidMotion


  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  
  template<typename M6,typename T>
  M6 rigidAction( const T& m, const M6 & v );

  template<>
  Motion6::R32 rigidAction( const RigidMotion::EigenM & m,const Motion6::R32 & v )
  {
    // R*(v-pxw),R*w
    return Motion6::R32 (m.linear()*(v.linear()-m.translation().cross(v.angular())),
			 m.linear()*v.angular()  );
    
    // Motion6::R32 res(v.linear()-m.translation().cross(v.angular()),
    // 		v.angular()  );
    // return Motion6::R32(m.linear()*res.base());
  }

  template<>
  Motion6::R32 rigidAction( const RigidMotion::Rp & m,const Motion6::R32 & v )
  {
    // R*(v-pxw),R*w
    //return Motion6::R32 (m.R*(v.linear()-m.p.cross(v.angular())),m.R*v.angular());
    Motion6::R32 res; res.base().noalias() = m.R*v.base();
    res.linear().noalias() -= m.p.cross( res.angular() );
    return res;
  }


  template<>
  Motion6::VW rigidAction( const RigidMotion::Rp & m,const Motion6::VW & v )
  {
    // R*(v-pxw),R*w
    return Motion6::VW (m.R*(v.v-m.p.cross(v.w)),m.R*v.w);
  }

  template<>
  Motion6::VW rigidAction( const RigidMotion::EigenM & m,const Motion6::VW & v )
  {
    // R*(v-pxw),R*w
    return Motion6::VW (m.linear()*(v.v-m.translation().cross(v.w)),m.linear()*v.w);
    //return Motion6::VW();
  }

  template<>
  Motion6::R32 rigidAction( const RigidMotion::Rx & m,const Motion6::R32 & v )
  {
    // R*(v-pxw),R*w
    Motion6::R32 cp = v,res;
    cp.linear() -= m.p.cross(v.angular());

    res.base().row(0) = cp.base().row(0);
    res.base().row(1) =   cp.base().row(1)*m.ca + cp.base().row(2)*m.sa ;
    res.base().row(1) = - cp.base().row(1)*m.sa + cp.base().row(2)*m.ca ;
    return res;
  }

  template<>
  Motion6::Simple32 rigidAction( const RigidMotion::Simple33 & m,const Motion6::Simple32 & v)
  {
    return (m*v).eval();
  }

  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------------------------------------------- */
  
  template<typename RigidMotion_t>
  double geom()
  {
    const int nbJoint = 36;
    std::vector< RigidMotion_t > sXp(nbJoint);
    for( int i=0; i<nbJoint; ++i )
      {
	sXp[i] = RigidMotion::random<RigidMotion_t>();
      }
    
    std::vector< RigidMotion_t > oXi(nbJoint);
    oXi[0] = RigidMotion::random<RigidMotion_t>();

    StackTicToc timer(StackTicToc::US);

    timer.tic();
    SMOOTH(1000)
      for( int i=1; i<nbJoint; ++i )
	{
	  oXi[i] = oXi[i-1]*sXp[i];
	}
    return timer.toc(StackTicToc::US)/1000;
    //timer.toc(std::cout,1000);
  }

  template<typename RigidMotion_t,typename Motion6_t>
  double kine()
  {
    const int nbJoint = 36;
    std::vector< RigidMotion_t > sXp(nbJoint);
    for( int i=0; i<nbJoint; ++i )
      {
	sXp[i] = RigidMotion::random<RigidMotion_t>();
      }
    
    std::vector< Motion6_t > vi(nbJoint);
    std::vector< Motion6_t > ai(nbJoint);
    std::vector< RigidMotion_t > oXi(nbJoint);
  
    vi[0] = Motion6::random<Motion6_t>();
    ai[0] = Motion6::random<Motion6_t>();
    oXi[0] = RigidMotion_t();
    StackTicToc timer(StackTicToc::US);

    timer.tic();
    SMOOTH(1000)
      for( int i=1; i<nbJoint; ++i )
	{
	  //oXi[i] = oXi[i-1]*sXp[i];
	  vi[i] = rigidAction(sXp[i],vi[i-1]);
	  //ai[i] = rigidAction(sXp[i],ai[i-1]);
	}
    return timer.toc(StackTicToc::US)/1000;
    //timer.toc(std::cout,1000);
  }

  template<typename RigidMotion_t,typename Motion6_t>
  double kinegeom()
  {
    const int nbJoint = 36;
    std::vector< RigidMotion_t > sXp(nbJoint);
    for( int i=0; i<nbJoint; ++i )
      {
	sXp[i] = RigidMotion::random<RigidMotion_t>();
      }
    
    std::vector< Motion6_t > vi(nbJoint);
    std::vector< Motion6_t > ai(nbJoint);
    std::vector< RigidMotion_t > oXi(nbJoint);
  
    vi[0] = Motion6::random<Motion6_t>();
    ai[0] = Motion6::random<Motion6_t>();
    oXi[0] = RigidMotion_t();
    StackTicToc timer(StackTicToc::US);

    timer.tic();
    SMOOTH(1000)
      for( int i=1; i<nbJoint; ++i )
	{
	  oXi[i] = oXi[i-1]*sXp[i];
	  vi[i] = rigidAction(sXp[i],vi[i-1]);
	  ai[i] = rigidAction(sXp[i],ai[i-1]);
	}
    return timer.toc(StackTicToc::US)/1000;
    //timer.toc(std::cout,1000);
  }

} // namespace se3


double testCos()
{
  std::vector<double> x(1000),cx(1000);
  for(int i=0;i<1000;++i) 
    { 
      x[i] = Eigen::Matrix<double,1,1>::Random()[0];
    }

    StackTicToc timer(StackTicToc::US);

    timer.tic();
    SMOOTH(1000)
      for( int i=1; i<1000; ++i )
	{
	  cx[i] = cos(x[i]);
	}
    return timer.toc(StackTicToc::US)/1000/1000;
}

/* ------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------------------- */
  
#include <map>
typedef double (*function_t)();
typedef std::pair<function_t,std::string> functiondoc_t;
std::map<std::string,functiondoc_t> fs;

void buildFsMap(bool verbose=false)
{
  fs["geom1"] = functiondoc_t(&se3::geom<se3::RigidMotion::Rp>,"Geometry with simple R+p");
  fs["geom2"] = functiondoc_t(&se3::geom<se3::RigidMotion::EigenM>,"Geometry with Eigen::Affine3d");
  fs["geomRx"] = functiondoc_t(&se3::geom<se3::RigidMotion::Rx>,"Geometry with Revolute x");
  fs["geomLowBound"] = functiondoc_t(&se3::geom<se3::RigidMotion::Simple33>,"Theoretial lower bound for geometry");

  fs["kine1"] = functiondoc_t(&se3::kine<se3::RigidMotion::Rp,    se3::Motion6::R32>,"Kine with SE3::R+p and M6::R32");
  fs["kine2"] = functiondoc_t(&se3::kine<se3::RigidMotion::EigenM,se3::Motion6::R32>,"Kine with SE3::Eigen and M6::R32");
  fs["kine3"] = functiondoc_t(&se3::kine<se3::RigidMotion::Rp,    se3::Motion6::VW>,"Kine with SE3::R+p and M6::VW");
  fs["kine4"] = functiondoc_t(&se3::kine<se3::RigidMotion::EigenM,se3::Motion6::VW>,"Kine with SE3::Eigen and M6::VW");
  fs["kineRx"] = functiondoc_t(&se3::kine<se3::RigidMotion::Rx,    se3::Motion6::R32>,"Kine with SE3::Rx and M6::R32");
  fs["kineLowBound"] = functiondoc_t(&se3::kine<se3::RigidMotion::Simple33,se3::Motion6::Simple32>,"Theoretial lower bound for kine");

  fs["kinegeom1"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::Rp,    se3::Motion6::R32>,"Kine-Geom with SE3::R+p and M6::R32");
  fs["kinegeom2"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::EigenM,se3::Motion6::R32>,"Kine-Geom with SE3::Eigen and M6::R32");
  fs["kinegeom3"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::Rp,    se3::Motion6::VW>,"Kine-Geom with SE3::R+p and M6::VW");
  fs["kinegeom4"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::EigenM,se3::Motion6::VW>,"Kine-Geom with SE3::Eigen and M6::VW");
  fs["kinegeomRx"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::Rx,    se3::Motion6::R32>,"Kine-Geom with SE3::Rx and M6::R32");
  fs["kinegeomLowBound"] = functiondoc_t(&se3::kinegeom<se3::RigidMotion::Simple33,se3::Motion6::Simple32>,"Theoretial lower bound for kine-geom");

  fs["tcos"] = functiondoc_t(&testCos,"test cos");

  if(verbose)
    {
      for( std::map<std::string,functiondoc_t>::iterator it=fs.begin();it!=fs.end();++it )
	std::cout << it->first << " ";
      std::cout << std::endl;
    }
}

