/*
 * Copyright 2026 INRIA
 */

#include <cstdint>
#include <iostream>
#include <sstream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/ufunc.hpp"
#include "eigenpy/user-type.hpp"
#include "eigenpy/user-type-traits.hpp"

// A scalar type whose all-zero byte pattern is *not* a valid value, mimicking
// pointer-backed types such as Boost.Multiprecision mpfr/mpc numbers (whose
// zero-filled state holds a null limb pointer and crashes the backend
// library when used). Here, instead of crashing, every read of a
// never-constructed slot increments a module-global violation counter that
// the Python test asserts to be zero.
struct GuardedScalar;

std::size_t& violation_count() {
  static std::size_t count = 0;
  return count;
}

struct GuardedScalar {
  static const std::uint64_t MAGIC = 0x5AFE5AFE5AFE5AFEull;

  GuardedScalar() : magic(MAGIC), value(0.) {}
  GuardedScalar(const double& v) : magic(MAGIC), value(v) {}
  GuardedScalar(const GuardedScalar& other)
      : magic(MAGIC), value(other.read()) {}

  // Assignment must tolerate a zeroed *this (the user_type_traits contract):
  // only the *source* is read.
  GuardedScalar& operator=(const GuardedScalar& other) {
    magic = MAGIC;
    value = other.read();
    return *this;
  }
  GuardedScalar& operator=(const double& v) {
    magic = MAGIC;
    value = v;
    return *this;
  }

  GuardedScalar operator+(const GuardedScalar& other) const {
    return GuardedScalar(read() + other.read());
  }
  GuardedScalar operator-(const GuardedScalar& other) const {
    return GuardedScalar(read() - other.read());
  }
  GuardedScalar operator*(const GuardedScalar& other) const {
    return GuardedScalar(read() * other.read());
  }
  GuardedScalar operator/(const GuardedScalar& other) const {
    return GuardedScalar(read() / other.read());
  }

  GuardedScalar& operator+=(const GuardedScalar& other) {
    value = read() + other.read();
    magic = MAGIC;
    return *this;
  }
  GuardedScalar& operator-=(const GuardedScalar& other) {
    value = read() - other.read();
    magic = MAGIC;
    return *this;
  }
  GuardedScalar& operator*=(const GuardedScalar& other) {
    value = read() * other.read();
    magic = MAGIC;
    return *this;
  }
  GuardedScalar& operator/=(const GuardedScalar& other) {
    value = read() / other.read();
    magic = MAGIC;
    return *this;
  }

  bool operator==(const GuardedScalar& other) const {
    return read() == other.read();
  }
  bool operator!=(const GuardedScalar& other) const {
    return read() != other.read();
  }
  bool operator<(const GuardedScalar& other) const {
    return read() < other.read();
  }
  bool operator<=(const GuardedScalar& other) const {
    return read() <= other.read();
  }
  bool operator>(const GuardedScalar& other) const {
    return read() > other.read();
  }
  bool operator>=(const GuardedScalar& other) const {
    return read() >= other.read();
  }

  GuardedScalar operator-() const { return GuardedScalar(-read()); }

  operator double() const { return read(); }

  std::string print() const {
    std::stringstream ss;
    ss << "value: " << read() << std::endl;
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const GuardedScalar& x) {
    os << x.read();
    return os;
  }

  // Every read path funnels through here: using a never-constructed slot
  // (magic == 0, the numpy zero-fill pattern) is recorded instead of
  // crashing, so the Python test can assert on it.
  double read() const {
    if (magic != MAGIC) ++violation_count();
    return value;
  }

  std::uint64_t magic;
  double value;
};

namespace eigenpy {
template <>
struct user_type_traits<GuardedScalar> {
  enum { requires_initialization = true };
  static bool is_uninitialized(const GuardedScalar& x) { return x.magic == 0; }
};
}  // namespace eigenpy

namespace Eigen {
template <>
struct NumTraits<GuardedScalar> {
  typedef GuardedScalar Real;
  typedef GuardedScalar NonInteger;
  typedef GuardedScalar Literal;
  typedef GuardedScalar Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 2,
    MulCost = 2
  };

  static GuardedScalar epsilon() {
    return GuardedScalar(std::numeric_limits<double>::epsilon());
  }
  static GuardedScalar dummy_precision() {
    return GuardedScalar(NumTraits<double>::dummy_precision());
  }
  static GuardedScalar highest() {
    return GuardedScalar(std::numeric_limits<double>::max());
  }
  static GuardedScalar lowest() {
    return GuardedScalar(std::numeric_limits<double>::min());
  }
  static int digits10() { return std::numeric_limits<double>::digits10; }
  static int max_digits10() {
    return std::numeric_limits<double>::max_digits10;
  }
};
}  // namespace Eigen

std::size_t get_violation_count() { return violation_count(); }
void reset_violation_count() { violation_count() = 0; }

double guarded_to_double(const GuardedScalar& x) { return x.read(); }

BOOST_PYTHON_MODULE(user_type_uninitialized) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  bp::class_<GuardedScalar>("GuardedScalar", bp::init<double>(bp::arg("value")))
      .def(bp::self + bp::self)
      .def(bp::self - bp::self)
      .def(bp::self * bp::self)
      .def(bp::self / bp::self)

      .def(bp::self += bp::self)
      .def(bp::self -= bp::self)
      .def(bp::self *= bp::self)
      .def(bp::self /= bp::self)

      .def("__repr__", &GuardedScalar::print)
      .def("__float__", &guarded_to_double, bp::arg("self"));

  eigenpy::registerNewType<GuardedScalar>();
  eigenpy::registerCommonUfunc<GuardedScalar>();

  typedef Eigen::Matrix<GuardedScalar, Eigen::Dynamic, Eigen::Dynamic>
      GuardedMatrix;
  eigenpy::EigenToPyConverter<GuardedMatrix>::registration();
  eigenpy::EigenFromPyConverter<GuardedMatrix>::registration();

  eigenpy::registerCast<GuardedScalar, double>(true);
  eigenpy::registerCast<double, GuardedScalar>(true);
  eigenpy::registerCast<GuardedScalar, int>(false);
  eigenpy::registerCast<int, GuardedScalar>(true);
  eigenpy::registerCast<GuardedScalar, long long>(false);
  eigenpy::registerCast<long long, GuardedScalar>(true);
  eigenpy::registerCast<GuardedScalar, long>(false);
  eigenpy::registerCast<long, GuardedScalar>(true);

  bp::implicitly_convertible<double, GuardedScalar>();
  bp::implicitly_convertible<GuardedScalar, double>();

  bp::def("get_violation_count", &get_violation_count,
          "Number of reads of never-constructed (zero-filled) slots so far.");
  bp::def("reset_violation_count", &reset_violation_count,
          "Reset the violation counter to zero.");
}
