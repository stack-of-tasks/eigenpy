/// Copyright 2023 LAAS-CNRS, INRIA
#include <eigenpy/eigenpy.hpp>

using std::shared_ptr;
namespace bp = boost::python;

// fwd declaration
struct MyVirtualData;

/// A virtual class with two pure virtual functions taking different signatures,
/// and a polymorphic factory function.
struct MyVirtualClass {
  MyVirtualClass() {}
  virtual ~MyVirtualClass() {}

  // polymorphic fn taking arg by shared_ptr
  virtual void doSomethingPtr(shared_ptr<MyVirtualData> const &data) const = 0;
  // polymorphic fn taking arg by reference
  virtual void doSomethingRef(MyVirtualData &data) const = 0;

  virtual shared_ptr<MyVirtualData> createData() const {
    return std::make_shared<MyVirtualData>(*this);
  }
};

struct MyVirtualData {
  MyVirtualData(MyVirtualClass const &) {}
  virtual ~MyVirtualData() {}  // virtual dtor to mark class as polymorphic
};

shared_ptr<MyVirtualData> callDoSomethingPtr(const MyVirtualClass &obj) {
  auto d = obj.createData();
  printf("Created MyVirtualData with address %p\n", (void *)d.get());
  obj.doSomethingPtr(d);
  return d;
}

shared_ptr<MyVirtualData> callDoSomethingRef(const MyVirtualClass &obj) {
  auto d = obj.createData();
  printf("Created MyVirtualData with address %p\n", (void *)d.get());
  obj.doSomethingRef(*d);
  return d;
}

void throw_virtual_not_implemented_error() {
  throw std::runtime_error("Called C++ virtual function.");
}

/// Wrapper classes
struct VirtualClassWrapper : MyVirtualClass, bp::wrapper<MyVirtualClass> {
  void doSomethingPtr(shared_ptr<MyVirtualData> const &data) const override {
    if (bp::override fo = this->get_override("doSomethingPtr")) {
      /// shared_ptr HAS to be passed by value.
      /// Boost.Python's argument converter has the wrong behaviour for
      /// reference_wrapper<shared_ptr<T>>, so boost::ref(data) does not work.
      fo(data);
      return;
    }
    throw_virtual_not_implemented_error();
  }

  /// The data object is passed by mutable reference to this function,
  /// and wrapped in a @c boost::reference_wrapper when passed to the override.
  /// Otherwise, Boost.Python's argument converter will convert to Python by
  /// value and create a copy.
  void doSomethingRef(MyVirtualData &data) const override {
    if (bp::override fo = this->get_override("doSomethingRef")) {
      fo(boost::ref(data));
      return;
    }
    throw_virtual_not_implemented_error();
  }

  shared_ptr<MyVirtualData> createData() const override {
    if (bp::override fo = this->get_override("createData")) {
      bp::object result = fo().as<bp::object>();
      return bp::extract<shared_ptr<MyVirtualData>>(result);
    }
    return default_createData();
  }

  shared_ptr<MyVirtualData> default_createData() const {
    return MyVirtualClass::createData();
  }
};

/// This "trampoline class" does nothing but is ABSOLUTELY required to ensure
/// downcasting works properly with non-smart ptr signatures. Otherwise,
/// there is no handle to the original Python object ( @c PyObject *).
/// Every single polymorphic type exposed to Python should be exposed through
/// such a trampoline. Users can also create their own wrapper classes by taking
/// inspiration from boost::python::wrapper<T>.
struct DataWrapper : MyVirtualData, bp::wrapper<MyVirtualData> {
  /// we have to use-declare non-defaulted constructors
  /// (see https://en.cppreference.com/w/cpp/language/default_constructor)
  /// or define them manually.
  using MyVirtualData::MyVirtualData;
};

/// Take and return a const reference
const MyVirtualData &iden_ref(const MyVirtualData &d) {
  // try cast to holder
  return d;
}

/// Take a shared_ptr (by const reference or value, doesn't matter), return by
/// const reference
const MyVirtualData &iden_shared(const shared_ptr<MyVirtualData> &d) {
  // get boost.python's custom deleter
  // boost.python hides the handle to the original object in there
  // dter being nonzero indicates shared_ptr was wrapped by Boost.Python
  auto *dter = std::get_deleter<bp::converter::shared_ptr_deleter>(d);
  if (dter != 0) printf("> input shared_ptr has a deleter\n");
  return *d;
}

/// Take and return a shared_ptr
shared_ptr<MyVirtualData> copy_shared(const shared_ptr<MyVirtualData> &d) {
  auto *dter = std::get_deleter<bp::converter::shared_ptr_deleter>(d);
  if (dter != 0) printf("> input shared_ptr has a deleter\n");
  return d;
}

BOOST_PYTHON_MODULE(bind_virtual_factory) {
  assert(std::is_polymorphic<MyVirtualClass>::value &&
         "MyVirtualClass should be polymorphic!");
  assert(std::is_polymorphic<MyVirtualData>::value &&
         "MyVirtualData should be polymorphic!");

  bp::class_<VirtualClassWrapper, boost::noncopyable>(
      "MyVirtualClass", bp::init<>(bp::args("self")))
      .def("doSomething", bp::pure_virtual(&MyVirtualClass::doSomethingPtr),
           bp::args("self", "data"))
      .def("doSomethingRef", bp::pure_virtual(&MyVirtualClass::doSomethingRef),
           bp::args("self", "data"))
      .def("createData", &MyVirtualClass::createData,
           &VirtualClassWrapper::default_createData, bp::args("self"));

  bp::register_ptr_to_python<shared_ptr<MyVirtualData>>();
  /// Trampoline used as 1st argument
  /// otherwise if passed as "HeldType", we need to define
  /// the constructor and call initializer manually.
  bp::class_<DataWrapper, boost::noncopyable>("MyVirtualData", bp::no_init)
      .def(bp::init<MyVirtualClass const &>(bp::args("self", "model")));

  bp::def("callDoSomethingPtr", callDoSomethingPtr, bp::args("obj"));
  bp::def("callDoSomethingRef", callDoSomethingRef, bp::args("obj"));

  bp::def("iden_ref", iden_ref, bp::return_internal_reference<>());
  bp::def("iden_shared", iden_shared, bp::return_internal_reference<>());
  bp::def("copy_shared", copy_shared);
}
