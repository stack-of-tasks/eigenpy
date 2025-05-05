#pragma once

#ifndef EXAMPLE_A_CLASS
#define EXAMPLE_A_CLASS

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/user-type.hpp>
#include <eigenpy/ufunc.hpp>

#include <boost/multiprecision/mpc.hpp>

#include <boost/multiprecision/eigen.hpp>



namespace bmp = boost::multiprecision;

using mpfr_float =
    boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<0>,
                                  boost::multiprecision::et_off>;

using bmp::backends::mpc_complex_backend;
using mpfr_complex =
    bmp::number<mpc_complex_backend<0>,
                bmp::et_off>;  // T is a variable-precision complex number with
                               // expression templates turned on.


class Whatevs : public boost::python::def_visitor<Whatevs>{

public:
	static
	void qwfp(mpfr_float const& c, Eigen::Matrix<mpfr_float,Eigen::Dynamic, Eigen::Dynamic> const& M){}
};

class JustSomeClass
{
public:
	JustSomeClass(){};
	~JustSomeClass() = default;

	void foo(mpfr_float const& the_constant, Eigen::Matrix<mpfr_float, Eigen::Dynamic, Eigen::Dynamic> const& M){};
	
	static int bar(JustSomeClass const& self, mpfr_float const& c, Eigen::Matrix<mpfr_float,Eigen::Dynamic, Eigen::Dynamic> const& M){return 42;}
};





void ExposeAClass();


#endif
