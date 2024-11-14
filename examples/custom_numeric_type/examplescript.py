import eigenpy_example_custom_numeric_type as example

x = example.MpfrComplex(2)  # the number 2, in variable precision as a complex number

import numpy as np

M = np.zeros((3,4),dtype=example.MpfrComplex)  # make an array of the custom numeric type

print(M)


example.set_to_ones(M)

print(M)