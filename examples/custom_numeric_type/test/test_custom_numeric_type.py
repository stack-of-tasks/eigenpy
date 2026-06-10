# pytest unit tests for the custom numeric type example.
# the custom numeric types are multi precision variable-precision floats and complex
# numbers from Boost.Multiprecision

# silviana amethyst
# Max Planck Institute of Molecular Cell Biology and Genetics
# fall 2024


import math
import sys

import pytest

sys.path.append("../build")

import eigenpy_example_custom_numeric_type as example
import numpy as np


@pytest.fixture(params=[example.MpfrFloat, example.MpfrComplex])
def dtype(request):
    return request.param


@pytest.fixture()
def empty_with_conversion(dtype):
    yield np.array(np.empty(3).astype(np.int64), dtype=dtype)


@pytest.fixture()
def zeros_with_conversion(dtype):
    yield np.array(np.zeros(3).astype(np.int64), dtype=dtype)


@pytest.fixture()
def ones_with_conversion(dtype):
    yield np.array(np.ones(3).astype(np.int64), dtype=dtype)


@pytest.fixture()
def empty_without_conversion(dtype):
    yield np.empty((3,), dtype=dtype)


@pytest.fixture()
def zeros_without_conversion(dtype):
    yield np.zeros((3,), dtype=dtype)


@pytest.fixture()
def ones_without_conversion(dtype):
    yield np.ones((3,), dtype=dtype)


class TestAllTypes:
    def test_make_empty_with_conversion(self, dtype, empty_with_conversion):
        pass

    def test_make_zeros_with_conversion(self, dtype, zeros_with_conversion):
        # A = np.array( np.zeros( (3)).astype(np.int32),dtype=dtype)  # make an array of the custom numeric type
        for x in zeros_with_conversion:
            assert x == 0

    def test_make_in_numpy_then_modify_in_cpp(self, dtype, zeros_with_conversion):
        A = zeros_with_conversion
        example.set_to_ones(A)

        assert A[0] == dtype(1)

    def test_make_in_cpp_then_modify_in_cpp_once(self, dtype):

        A = example.make_a_vector_in_cpp(
            4, dtype(1)
        )  # the second argument is used only for type dispatch
        example.set_to_ones(A)

        for a in A:
            assert a == dtype(1)

    def test_make_in_cpp_then_modify_in_cpp_list(self, dtype):

        my_list = []

        for ii in range(10):
            A = example.make_a_vector_in_cpp(
                4, dtype(1)
            )  # the second argument is used only for type dispatch
            my_list.append(A)

        for A in my_list:
            example.set_to_ones(A)
            for a in A:
                assert a == dtype(1)

            example.set_to_ones(A)

    def test_make_then_call_function_taking_scalar_and_vector(
        self, dtype, zeros_with_conversion
    ):
        A = zeros_with_conversion
        s = dtype(3)

        example.a_function_taking_both_a_scalar_and_a_vector(s, A)

    def test_set_entire_array_to_one_value(self, dtype):
        A = example.make_a_vector_in_cpp(
            10, dtype(0)
        )  # again, type dispatch on the second

        cst = dtype("13") / dtype("7")  # 13/7 seems like a good number.  why not.

        example.set_all_entries_to_constant(
            A, cst
        )  # all entries should become the constant, in this case 13

    def test_class_function_with_both_arguments(self, dtype):
        dtype = example.MpfrFloat

        c = example.JustSomeClass()

        A = example.make_a_vector_in_cpp(
            10, dtype(0)
        )  # again, type dispatch on the second

        cst = dtype("13") / dtype("7")  # 13/7 seems like a good number.  why not.

        c.foo(cst, A)  # all entries should become the constant, in this case 13
        example.qwfp(cst, A)

    def test_numpy_norm(self, dtype, ones_with_conversion):
        if dtype is example.MpfrComplex:
            pytest.xfail(
                "np.linalg.norm of a complex custom dtype needs ufunc loops "
                "(sqrt/absolute) that are not registered"
            )
        A = ones_with_conversion
        assert np.abs(np.linalg.norm(A) - np.sqrt(3)) < 1e-10

    def test_numpy_manual_norm(self, ones_with_conversion):
        A = ones_with_conversion
        # np.sqrt/np.absolute have no ufunc loop for the custom dtype, so take
        # the sum as a scalar and use math.sqrt on its float value (the squared
        # norm of a vector of ones is real; multiply *is* registered)
        s = np.sum(A * A)
        if isinstance(s, example.MpfrComplex):
            s = s.real
        norm = math.sqrt(float(s))
        assert abs(norm - math.sqrt(3)) < 1e-10


# Everything below used to crash inside libmpfr before eigenpy's
# user_type_traits guards: numpy zero-fills fresh buffers (np.zeros, np.empty,
# ufunc outputs) without running a C++ constructor, and an all-zero mpfr/mpc
# is Boost.Multiprecision's *uninitialized* state (null limb pointer), not a
# valid zero. eigenpy now treats such slots as an exact 0 on read.
class TestPreviouslyCrashingCreation:
    def test_empty_getitem_heals(self, dtype, empty_without_conversion):
        for x in empty_without_conversion:
            assert x == dtype(0)

    def test_zeros_getitem(self, dtype, zeros_without_conversion):
        for x in zeros_without_conversion:
            assert x == dtype(0)

    def test_repr_of_unwritten(self, dtype):
        repr(np.empty((3,), dtype=dtype))

    def test_add_unwritten(self, dtype, zeros_without_conversion):
        S = zeros_without_conversion + np.zeros((3,), dtype=dtype)
        for x in S:
            assert x == dtype(0)

    def test_add_mixed_written_unwritten(self, dtype, zeros_without_conversion):
        W = example.make_a_vector_in_cpp(3, dtype(1))
        example.set_to_ones(W)
        S = W + zeros_without_conversion
        for x in S:
            assert x == dtype(1)

    def test_dot_of_unwritten(self, dtype, zeros_without_conversion):
        Z = zeros_without_conversion
        assert np.dot(Z, Z) == dtype(0)
        assert np.inner(Z, Z) == dtype(0)

    def test_matmul_of_unwritten(self, dtype):
        Z = np.zeros((3, 3), dtype=dtype)
        P = Z @ Z
        assert P[0, 0] == dtype(0)

    def test_dot_of_written(self, dtype):
        A = np.empty((3,), dtype=dtype)
        for i in range(3):
            A[i] = dtype(i + 1)
        assert np.dot(A, A) == dtype(1 + 4 + 9)

    def test_dot_conjugates_like_eigen(self):
        # eigenpy's dotfunc keeps Eigen's dot semantics: the first operand is
        # conjugated. For c = i, conj(c) * c = 1.
        c = example.MpfrComplex(0, 1)
        A = np.empty((1,), dtype=example.MpfrComplex)
        A[0] = c
        assert np.dot(A, A) == example.MpfrComplex(1, 0)

    def test_full(self, dtype):
        F = np.full((3,), dtype(7))
        for x in F:
            assert x == dtype(7)

    def test_copy_of_unwritten(self, dtype):
        C = np.zeros((3,), dtype=dtype).copy()
        for x in C:
            assert x == dtype(0)

    def test_noncontiguous_copy(self, dtype):
        C = np.zeros((6,), dtype=dtype)[::2].copy()
        for x in C:
            assert x == dtype(0)

    def test_astype_to_float(self, dtype):
        if dtype is example.MpfrComplex:
            pytest.skip("the complex type only registers casts from integer types")
        as_float = np.zeros((3,), dtype=dtype).astype(np.float64)
        assert (as_float == 0.0).all()

    def test_astype_from_int(self, dtype):
        from_int = np.zeros((3,), dtype=np.int64).astype(dtype)
        assert from_int[0] == dtype(0)

    def test_count_nonzero(self, dtype):
        Z = np.zeros((3,), dtype=dtype)
        assert np.count_nonzero(Z) == 0
        Z[1] = dtype(1)
        assert np.count_nonzero(Z) == 1
