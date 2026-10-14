# some code that exercises C++-exposed code using custom numeric type via EigenPy.

import math
import sys

sys.path.append("./")

import eigenpy_example_custom_numeric_type as example
import numpy as np


def make_empty_with_conversion(num_type):
    return np.array(np.empty(3).astype(np.int64), dtype=num_type)


def make_zeros_with_conversion(num_type):
    return np.array(
        np.zeros(3).astype(np.int32), dtype=num_type
    )  # make an array of the custom numeric type


def make_in_numpy_then_modify_in_cpp(num_type):
    A = make_zeros_with_conversion(num_type)
    example.set_to_ones(A)

    assert A[0] == num_type(1)


def make_in_cpp_then_modify_in_cpp_once(num_type):

    A = example.make_a_vector_in_cpp(
        4, num_type(1)
    )  # the second argument is used only for type dispatch
    example.set_to_ones(A)

    for a in A:
        assert a == num_type(1)


def make_in_cpp_then_modify_in_cpp_list(num_type):

    my_list = []

    for ii in range(10):
        A = example.make_a_vector_in_cpp(
            4, num_type(1)
        )  # the second argument is used only for type dispatch
        my_list.append(A)

    for A in my_list:
        example.set_to_ones(A)
        for a in A:
            assert a == num_type(1)

        example.set_to_ones(A)


def make_then_call_function_taking_scalar_and_vector(num_type):
    A = make_zeros_with_conversion(num_type)
    s = num_type(3)

    example.a_function_taking_both_a_scalar_and_a_vector(s, A)


def set_entire_array_to_one_value(num_type):
    A = example.make_a_vector_in_cpp(
        10, num_type(0)
    )  # again, type dispatch on the second

    cst = num_type("13") / num_type("7")  # 13/7 seems like a good number.  why not.

    example.set_all_entries_to_constant(
        A, cst
    )  # all entries should become the constant, in this case 13


def class_function_with_both_arguments():
    num_type = example.MpfrFloat

    c = example.JustSomeClass()

    A = example.make_a_vector_in_cpp(
        10, num_type(0)
    )  # again, type dispatch on the second

    cst = num_type("13") / num_type("7")  # 13/7 seems like a good number.  why not.

    c.foo(cst, A)  # all entries should become the constant, in this case 13
    example.qwfp(cst, A)


def numpy_norm(num_type):
    A = make_zeros_with_conversion(num_type)
    example.set_to_ones(A)

    # assert np.abs(np.linalg.norm(A) - np.sqrt(3)) < 1e-10


def numpy_manual_norm(num_type):
    A = make_zeros_with_conversion(num_type)
    example.set_to_ones(A)
    # np.sqrt/np.absolute have no ufunc loop registered for the custom dtype,
    # so take the sum as a scalar and use math.sqrt on its float value (the
    # squared norm of a vector of ones is real).
    s = np.sum(A * A)
    if isinstance(s, example.MpfrComplex):
        s = s.real
    norm = math.sqrt(float(s))
    assert abs(norm - math.sqrt(3)) < 1e-10


def expected_to_succeed(num_type):

    print(f"testing {num_type} at precision {num_type.default_precision()}")

    make_empty_with_conversion(num_type)
    make_zeros_with_conversion(num_type)

    make_in_numpy_then_modify_in_cpp(num_type)
    make_in_cpp_then_modify_in_cpp_once(num_type)
    make_in_cpp_then_modify_in_cpp_list(num_type)

    set_entire_array_to_one_value(num_type)

    make_then_call_function_taking_scalar_and_vector(num_type)

    class_function_with_both_arguments()

    numpy_norm(num_type)
    numpy_manual_norm(num_type)


def make_empty_without_conversion(num_type):
    A = np.empty((3), dtype=num_type)
    assert A[0] == num_type(0)  # never-written slots read as exact zero

    return A


def make_zeros_without_conversion(num_type):

    A = np.zeros((3), dtype=num_type)  # make an array of the custom numeric type
    assert A[0] == num_type(0)

    return A


def previously_crashed_now_works(num_type):
    # Direct creation of numpy arrays of the custom type (no conversion from a
    # built-in dtype) used to crash inside libmpfr: numpy zero-fills fresh
    # buffers but never runs a C++ constructor, and an all-zero mpfr/mpc is
    # Boost.Multiprecision's *uninitialized* state (null limb pointer), not a
    # valid zero. eigenpy::user_type_traits (specialized in header.hpp) now
    # tells eigenpy to treat such slots as an exact 0 on read.
    make_empty_without_conversion(num_type)
    Z = make_zeros_without_conversion(num_type)

    # arithmetic, dot products, casts and fills on fresh arrays all used to
    # reach the uninitialized slots too
    S = Z + Z
    assert S[0] == num_type(0)

    d = np.dot(Z, Z)
    assert d == num_type(0)

    F = np.full((3,), num_type(7))
    assert F[2] == num_type(7)

    # the complex type only registers casts *from* integer types, so the
    # cast-loop check (astype) is done with the real type only
    if num_type is example.MpfrFloat:
        as_float = Z.astype(np.float64)
        assert as_float[0] == 0.0
    from_int = np.zeros((3,), dtype=np.int64).astype(num_type)
    assert from_int[0] == num_type(0)


for prec in [20, 50, 100]:
    example.MpfrFloat.default_precision(prec)
    expected_to_succeed(example.MpfrFloat)

    example.MpfrComplex.default_precision(prec)
    expected_to_succeed(example.MpfrComplex)


# these used to crash (see the note in previously_crashed_now_works); they are
# fixed by eigenpy's user_type_traits guards.  2026.06
previously_crashed_now_works(example.MpfrFloat)
previously_crashed_now_works(example.MpfrComplex)

print("examplescript completed successfully")
