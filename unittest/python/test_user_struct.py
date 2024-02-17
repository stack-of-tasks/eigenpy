import numpy as np
from user_struct import MyStruct

x = np.ones(3)
y = np.ones(4)
ms = MyStruct(x, y)
print(ms.x)
print(ms.y)

ms.x[0] = 0.0

ms.x = x  # ok
assert np.allclose(ms.x, x)

ms.y[:] = y
ms.y = y  # segfault
