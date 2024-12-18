#include "a_class.hpp"


namespace bp = boost::python;

void ExposeAClass(){
	boost::python::class_<JustSomeClass>("JustSomeClass", "")

	.def("foo", &JustSomeClass::foo)
	.def("bar", &JustSomeClass::bar)
	;

	bp::def("qwfp", &Whatevs::qwfp);
}

