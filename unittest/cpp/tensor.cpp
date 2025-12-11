/*
 * Copyright 2023 INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"
namespace bp = boost::python;

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vector1x1(const Scalar& value) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ReturnType;
  return ReturnType::Constant(1, value);
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix1x1(
    const Scalar& value) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ReturnType;
  return ReturnType::Constant(1, 1, value);
}

template <typename Tensor>
Eigen::TensorRef<Tensor> make_ref(Tensor& tensor) {
  return Eigen::TensorRef<Tensor>(tensor);
}

template <typename Tensor>
void fill(Eigen::TensorRef<Tensor> tensor, typename Tensor::Scalar value) {
  for (Eigen::DenseIndex k = 0; k < tensor.size(); ++k)
    tensor.coeffRef(k) = value;
}

template <typename Tensor>
void print(const Tensor& tensor) {
  std::cout << tensor << std::endl;
}

template <typename Tensor>
void print_ref(const Eigen::TensorRef<const Tensor> tensor) {
  ::print(tensor);
}

template <typename Tensor>
void print_base(const Eigen::TensorBase<Tensor>& tensor) {
  ::print(tensor);
}

template <typename Tensor>
Tensor copy(const Eigen::TensorBase<Tensor>& tensor) {
  return const_cast<Tensor&>(static_cast<const Tensor&>(tensor));
}

template <typename Tensor>
Eigen::TensorRef<Tensor> ref(Eigen::TensorRef<Tensor> tensor) {
  return tensor;
}

template <typename Tensor>
const Eigen::TensorRef<const Tensor> const_ref(
    const Eigen::TensorRef<const Tensor> tensor) {
  return tensor;
}

template <typename Scalar, int Rank>
Eigen::Tensor<Scalar, Rank> emptyTensor() {
  return Eigen::Tensor<Scalar, Rank>();
}

template <typename Scalar>
Eigen::Tensor<Scalar, 1> zeroTensor1(const Eigen::DenseIndex r) {
  Eigen::Tensor<Scalar, 1> tensor(r);
  tensor.setZero();
  return tensor;
}

template <typename Scalar>
Eigen::Tensor<Scalar, 2> zeroTensor2(const Eigen::DenseIndex r,
                                     const Eigen::DenseIndex s) {
  Eigen::Tensor<Scalar, 2> tensor(r, s);
  tensor.setZero();
  return tensor;
}

template <typename Scalar>
Eigen::Tensor<Scalar, 3> zeroTensor3(const Eigen::DenseIndex r,
                                     const Eigen::DenseIndex s,
                                     const Eigen::DenseIndex t) {
  Eigen::Tensor<Scalar, 3> tensor(r, s, t);
  tensor.setZero();
  return tensor;
}

template <typename Scalar>
Eigen::Tensor<Scalar, 1> createTensor1(const Eigen::DenseIndex r,
                                       Scalar value) {
  Eigen::Tensor<Scalar, 1> tensor(r);
  fill(make_ref(tensor), value);
  return tensor;
}

template <typename Scalar>
Eigen::Tensor<Scalar, 2> createTensor2(const Eigen::DenseIndex r,
                                       const Eigen::DenseIndex s,
                                       Scalar value) {
  Eigen::Tensor<Scalar, 2> tensor(r, s);
  fill(make_ref(tensor), value);
  return tensor;
}

template <typename Scalar>
Eigen::Tensor<Scalar, 3> createTensor3(const Eigen::DenseIndex r,
                                       const Eigen::DenseIndex s,
                                       const Eigen::DenseIndex t,
                                       Scalar value) {
  Eigen::Tensor<Scalar, 3> tensor(r, s, t);
  fill(make_ref(tensor), value);
  return tensor;
}

template <typename Scalar, int Rank>
struct TensorContainer {
  typedef Eigen::Tensor<Scalar, Rank> Tensor;
  typedef Eigen::TensorRef<Tensor> TensorRef;
  typedef Eigen::Matrix<typename Tensor::Index, Rank, 1> Dimensions;

  Tensor m_tensor;
  TensorContainer(const Dimensions& dims) {
    typedef Eigen::array<typename Tensor::Index, Rank> InternalDimension;
    InternalDimension _dims;
    for (size_t k = 0; k < Rank; ++k) _dims[k] = dims[Eigen::DenseIndex(k)];

    m_tensor = Tensor(_dims);
  }

  Tensor get_copy() const { return m_tensor; }
  TensorRef get_ref() { return TensorRef(m_tensor); }
};

template <typename Scalar, int Rank>
void exposeTensorContainer() {
  typedef TensorContainer<Scalar, Rank> T;
  const std::string class_name = "TensorContainer" + std::to_string(Rank);
  bp::class_<T>(class_name.c_str(), bp::no_init)
      .def(bp::init<typename T::Dimensions>())
      .def("get_copy", &T::get_copy)
      .def("get_ref", &T::get_ref,
           bp::with_custodian_and_ward_postcall<0, 1>());
}

BOOST_PYTHON_MODULE(tensor) {
  using namespace Eigen;
  eigenpy::enableEigenPy();

  typedef Eigen::Tensor<double, 1> Tensor1;
  typedef Eigen::Tensor<double, 2> Tensor2;
  typedef Eigen::Tensor<double, 3> Tensor3;

  bp::def("emptyTensor1", emptyTensor<double, 1>);
  bp::def("emptyTensor2", emptyTensor<double, 2>);
  bp::def("emptyTensor3", emptyTensor<double, 3>);

  bp::def("zeroTensor1", zeroTensor1<double>);
  bp::def("zeroTensor2", zeroTensor2<double>);
  bp::def("zeroTensor3", zeroTensor3<double>);

  bp::def("createTensor1", createTensor1<double>);
  bp::def("createTensor2", createTensor2<double>);
  bp::def("createTensor3", createTensor3<double>);

  bp::def("print", print<Tensor1>);
  bp::def("print", print<Tensor2>);
  bp::def("print", print<Tensor3>);

  bp::def("print_ref", print_ref<Tensor1>);
  bp::def("print_ref", print_ref<Tensor2>);
  bp::def("print_ref", print_ref<Tensor3>);

  bp::def("print_base", print_base<Tensor1>);
  bp::def("print_base", print_base<Tensor2>);
  bp::def("print_base", print_base<Tensor3>);

  bp::def("copy", copy<Tensor1>);
  bp::def("copy", copy<Tensor2>);
  bp::def("copy", copy<Tensor3>);

  bp::def("fill", fill<Tensor1>);
  bp::def("fill", fill<Tensor2>);
  bp::def("fill", fill<Tensor3>);

  bp::def("ref", ref<Tensor1>, bp::with_custodian_and_ward_postcall<0, 1>());
  bp::def("ref", ref<Tensor2>, bp::with_custodian_and_ward_postcall<0, 1>());
  bp::def("ref", ref<Tensor3>, bp::with_custodian_and_ward_postcall<0, 1>());

  bp::def("const_ref", const_ref<Tensor1>,
          bp::with_custodian_and_ward_postcall<0, 1>());
  bp::def("const_ref", const_ref<Tensor2>,
          bp::with_custodian_and_ward_postcall<0, 1>());
  bp::def("const_ref", const_ref<Tensor3>,
          bp::with_custodian_and_ward_postcall<0, 1>());

  exposeTensorContainer<double, 1>();
  exposeTensorContainer<double, 2>();
  exposeTensorContainer<double, 3>();
}
