import numpy as np
import tensor

dim = np.array([10, 20, 30], dtype=np.int64)
t = tensor.TensorContainer3(dim)
r = t.get_ref()
r[:] = 0.0
c = t.get_copy()
r2 = tensor.ref(r)
cr = tensor.const_ref(r)
c2 = tensor.copy(cr)

assert np.all(c == r)
assert np.all(r2 == r)
assert np.all(cr == r)
assert np.all(c2 == r)

tensor.print_base(cr)
tensor.print_ref(cr)
tensor.print(cr)

r2[:] = 100.0
assert not np.all(c == r)
assert not np.all(c2 == r)
assert np.all(r2 == r)
assert np.all(cr == r)

tensor.print_base(cr)
tensor.print_ref(cr)
tensor.print(cr)
