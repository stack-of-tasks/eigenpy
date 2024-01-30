/// @file
/// @copyright Copyright 2024 CNRS INRIA

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/variant.hpp>

namespace bp = boost::python;

struct V1 {
  int v;
};
struct V2 {
  char v;
};
typedef boost::variant<V1, V2> MyVariant;

MyVariant make_variant() { return V1(); }

struct VariantHolder {
  MyVariant variant;
};

BOOST_PYTHON_MODULE(boost_variant) {
  using namespace eigenpy;

  enableEigenPy();

  bp::class_<V1>("V1", bp::init<>()).def_readwrite("v", &V1::v);
  bp::class_<V2>("V2", bp::init<>()).def_readwrite("v", &V2::v);

  typedef eigenpy::VariantConverter<MyVariant> Converter;
  Converter::registration();

  bp::def("make_variant", make_variant);

  boost::python::class_<VariantHolder>("VariantHolder", bp::init<>())
      .add_property("variant",
                    bp::make_getter(&VariantHolder::variant,
                                    Converter::return_internal_reference()),
                    bp::make_setter(&VariantHolder::variant));
}
