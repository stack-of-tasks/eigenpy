#include "eigenpy/eigenpy.hpp"
#include "eigenpy/optional.hpp"

#define OPTIONAL boost::optional
#define OPT_NONE boost::none

using opt_dbl = OPTIONAL<double>;

struct mystruct {
  OPTIONAL<int> a;
  opt_dbl b;
  OPTIONAL<std::string> msg{"i am struct"};
  mystruct() : a(OPT_NONE), b(boost::none) {}
  mystruct(int a, const opt_dbl &b = OPT_NONE) : a(a), b(b) {}
};

OPTIONAL<int> none_if_zero(int i) {
  if (i == 0)
    return OPT_NONE;
  else
    return i;
}

OPTIONAL<mystruct> create_if_true(bool flag, opt_dbl b = OPT_NONE) {
  if (flag) {
    return mystruct(0, b);
  } else {
    return OPT_NONE;
  }
}

OPTIONAL<Eigen::MatrixXd> random_mat_if_true(bool flag) {
  if (flag)
    return Eigen::MatrixXd(Eigen::MatrixXd::Random(4, 4));
  else
    return OPT_NONE;
}

BOOST_PYTHON_MODULE(bind_optional) {
  using namespace eigenpy;
  OptionalConverter<int>::registration();
  OptionalConverter<double>::registration();
  OptionalConverter<std::string>::registration();
  OptionalConverter<mystruct>::registration();
  OptionalConverter<Eigen::MatrixXd>::registration();
  enableEigenPy();

  bp::class_<mystruct>("mystruct", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def(bp::init<int, bp::optional<const opt_dbl &> >(
          bp::args("self", "a", "b")))
      .add_property(
          "a",
          bp::make_getter(&mystruct::a,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::a))
      .add_property(
          "b",
          bp::make_getter(&mystruct::b,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::b))
      .add_property(
          "msg",
          bp::make_getter(&mystruct::msg,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&mystruct::msg));

  bp::def("none_if_zero", none_if_zero, bp::args("i"));
  bp::def("create_if_true", create_if_true, bp::args("flag", "b"));
  bp::def("random_mat_if_true", random_mat_if_true, bp::args("flag"));
}
