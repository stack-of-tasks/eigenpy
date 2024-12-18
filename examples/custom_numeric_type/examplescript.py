# some code that exercises C++-exposed code using custom numeric type via EigenPy.

import sys

sys.path.append('./')

import numpy as np
import eigenpy_example_custom_numeric_type as example


def make_empty_with_conversion(num_type):
	return np.array( np.empty( (3)).astype(np.int64),dtype=num_type)


def make_zeros_with_conversion(num_type):
	return np.array( np.zeros( (3)).astype(np.int32),dtype=num_type)  # make an array of the custom numeric type








def make_in_numpy_then_modify_in_cpp(num_type):
	A = make_zeros_with_conversion(num_type)
	example.set_to_ones(A)

	assert(A[0] == num_type(1))

def make_in_cpp_then_modify_in_cpp_once(num_type):

	A = example.make_a_vector_in_cpp(4,num_type(1)) # the second argument is used only for type dispatch
	example.set_to_ones(A)

	for a in A:
		assert(a == num_type(1))


def make_in_cpp_then_modify_in_cpp_list(num_type):

	my_list = []

	for ii in range(10):
		A = example.make_a_vector_in_cpp(4,num_type(1)) # the second argument is used only for type dispatch
		my_list.append( A ) 
	
	for A in my_list:
		example.set_to_ones(A)
		for a in A:
			assert(a == num_type(1))

		example.set_to_ones(A)


def make_then_call_function_taking_scalar_and_vector(num_type):
	A = make_zeros_with_conversion(num_type)
	s = num_type(3)

	result = example.a_function_taking_both_a_scalar_and_a_vector(s, A)



def set_entire_array_to_one_value(num_type):
	A = example.make_a_vector_in_cpp(10, num_type(0)) # again, type dispatch on the second

	cst = num_type("13") / num_type("7") # 13/7 seems like a good number.  why not.

	example.set_all_entries_to_constant(A,cst)  # all entries should become the constant, in this case 13




def class_function_with_both_arguments():
	num_type = example.MpfrFloat

	c = example.JustSomeClass();

	A = example.make_a_vector_in_cpp(10, num_type(0)) # again, type dispatch on the second

	cst = num_type("13") / num_type("7") # 13/7 seems like a good number.  why not.

	c.foo(cst,A)  # all entries should become the constant, in this case 13
	example.qwfp(cst,A)



def numpy_norm(num_type):
	A = make_zeros_with_conversion(num_type)
	example.set_to_ones(A)

	# assert np.abs(np.linalg.norm(A) - np.sqrt(3)) < 1e-10


def numpy_manual_norm(num_type):
	A = make_zeros_with_conversion(num_type)
	example.set_to_ones(A)
	assert np.sqrt(np.sum((A)**2)) < 1e-10
	print('arst')



def expected_to_succeed(num_type):

	print(f'testing {num_type} at precision {num_type.default_precision()}') 

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
	return np.empty( (3),dtype=num_type)

def make_zeros_without_conversion(num_type):

	A = np.zeros( (3),dtype=num_type)  # make an array of the custom numeric type
	assert(A[0] == num_type(0))

	return A


def expected_to_crash(num_type):
	print("the following calls are expected to crash, not because they should, but because for whatever reason, eigenpy does not let us directly make numpy arrays WITHOUT converting")
	make_empty_without_conversion(num_type)
	make_zeros_without_conversion(num_type)











for prec in [20, 50, 100]:
	example.MpfrFloat.default_precision(prec)
	expected_to_succeed(example.MpfrFloat)

	example.MpfrComplex.default_precision(prec)
	expected_to_succeed(example.MpfrComplex)




# these really shouldn't crash!!!!  but they do, and it's a problem.  2024.12.18
expected_to_crash(example.MpfrFloat)
expected_to_crash(example.MpfrComplex)