# pytest unit tests for the custom numeric type example.
# the custom numeric types are multi precision variable-precision floats and complex
# numbers from Boost.Multiprecision

# silviana amethyst
# Max Planck Institute of Molecular Cell Biology and Genetics
# fall 2024


import pytest

import sys

sys.path.append("../build")

import numpy as np
import eigenpy_example_custom_numeric_type as example


@pytest.fixture(params=[example.MpfrFloat, example.MpfrComplex])
def dtype(request):
    return request.param


@pytest.fixture()
def empty_with_conversion(dtype):
    yield np.array( np.empty( (3)).astype(np.int64),dtype=dtype)


@pytest.fixture()
def zeros_with_conversion(dtype):
    yield np.array( np.zeros( (3)).astype(np.int64),dtype=dtype)

@pytest.fixture()
def ones_with_conversion(dtype):
    yield np.array( np.ones( (3)).astype(np.int64),dtype=dtype)









@pytest.fixture()
def empty_without_conversion(dtype):
    yield np.empty( (3,),dtype=dtype)


@pytest.fixture()
def zeros_without_conversion(dtype):
    yield np.zeros( (3,),dtype=dtype)

@pytest.fixture()
def ones_without_conversion(dtype):
    yield np.ones( (3,),dtype=dtype)







class TestAllTypes:


    def test_make_empty_with_conversion(self, dtype, empty_with_conversion):
        pass


    def test_make_zeros_with_conversion(self,dtype, zeros_with_conversion):
      # A = np.array( np.zeros( (3)).astype(np.int32),dtype=dtype)  # make an array of the custom numeric type
      for x in zeros_with_conversion:
        assert x == 0


    def test_make_in_numpy_then_modify_in_cpp(self,dtype, zeros_with_conversion):
        A = zeros_with_conversion
        example.set_to_ones(A)

        assert(A[0] == dtype(1))


    def test_make_in_cpp_then_modify_in_cpp_once(self,dtype):

        A = example.make_a_vector_in_cpp(4,dtype(1)) # the second argument is used only for type dispatch
        example.set_to_ones(A)

        for a in A:
            assert(a == dtype(1))


    def test_make_in_cpp_then_modify_in_cpp_list(self,dtype):

        my_list = []

        for ii in range(10):
            A = example.make_a_vector_in_cpp(4,dtype(1)) # the second argument is used only for type dispatch
            my_list.append( A ) 
        
        for A in my_list:
            example.set_to_ones(A)
            for a in A:
                assert(a == dtype(1))

            example.set_to_ones(A)


    def test_make_then_call_function_taking_scalar_and_vector(self,dtype, zeros_with_conversion):
        A = zeros_with_conversion
        s = dtype(3)

        result = example.a_function_taking_both_a_scalar_and_a_vector(s, A)



    def test_set_entire_array_to_one_value(self,dtype):
        A = example.make_a_vector_in_cpp(10, dtype(0)) # again, type dispatch on the second

        cst = dtype("13") / dtype("7") # 13/7 seems like a good number.  why not.

        example.set_all_entries_to_constant(A,cst)  # all entries should become the constant, in this case 13




    def test_class_function_with_both_arguments(self,dtype):
        dtype = example.MpfrFloat

        c = example.JustSomeClass();

        A = example.make_a_vector_in_cpp(10, dtype(0)) # again, type dispatch on the second

        cst = dtype("13") / dtype("7") # 13/7 seems like a good number.  why not.

        c.foo(cst,A)  # all entries should become the constant, in this case 13
        example.qwfp(cst,A)



    def test_numpy_norm(self,dtype, ones_with_conversion):
        A = ones_with_conversion
        # this fails for complexes because the linear algebra norm function doesn't know how to deal with the custom type
        assert np.abs(np.linalg.norm(A) - np.sqrt(3)) < 1e-10


    def test_numpy_manual_norm(self, ones_with_conversion):
        A = ones_with_conversion
        # this fails because the ufunc 'square' is not defined by Eigenpy.  
        assert np.sqrt(np.sum((A)**2)) < 1e-10





    
