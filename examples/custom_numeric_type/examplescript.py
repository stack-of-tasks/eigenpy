import sys

sys.path.append('./')

import eigenpy_example_custom_numeric_type as example



def try_things(num_type):

	print(f'testing {num_type}')

	x = num_type(2)  # the number 2, in variable precision as a complex number

	import numpy as np

	print('making array from empty WITH conversion')
	A = np.array( np.empty( (3,4)).astype(np.int64),dtype=num_type)

	print(A)

	print('making array from zeros WITH conversion')
	M = np.array( np.zeros( (3,4)).astype(np.int64),dtype=num_type)  # make an array of the custom numeric type

	print(M)

	assert(M[0,0] == num_type(0))


	example.set_to_ones(M)


	assert(M[0,0] == num_type(1))


	print(M)

	print('making zeros without conversion')
	B = np.zeros( (4,5), dtype=num_type)
	print(B)


try_things(example.MpfrFloat)
try_things(example.MpfrComplex)