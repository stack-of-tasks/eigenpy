import std_array

ints = std_array.get_arr_3_ints()
print(ints[0])
print(ints[1])
print(ints[2])
print(ints.tolist())
assert ints.tolist() == [1, 2, 3]

_ints_slice = ints[1:3]
print("Printing slice...")
for el in _ints_slice:
    print(el)

ref = [1, 2, 3]
assert len(ref[1:2]) == 1  # sanity check

assert len(_ints_slice) == 2, f"Slice size should be 1, got {len(_ints_slice)}"
assert _ints_slice[0] == 2
assert _ints_slice[1] == 3

# Test that insert/delete is impossible with the slice operator

# prepend
try:
    ints[0:0] = [0, 1]
except NotImplementedError:
    pass
else:
    assert False, "Insert value with slice operator should be impossible"

# append
try:
    ints[10:12] = [0]
except NotImplementedError:
    pass
else:
    assert False, "Insert value with slice operator should be impossible"

# append
try:
    ints[3:3] = [0]
except NotImplementedError:
    pass
else:
    assert False, "Insert value with slice operator should be impossible"

# Erase two elements and replace by one
try:
    ints[1:3] = [0]
except NotImplementedError:
    pass
else:
    assert False, "Insert value with slice operator should be impossible"

# Erase two elements and replace by three
try:
    ints[1:3] = [0, 1, 2]
except NotImplementedError:
    pass
else:
    assert False, "Insert value with slice operator should be impossible"

# Test that delete operator is not implemented
# Index delete
try:
    del ints[0]
except NotImplementedError:
    pass
else:
    assert False, "del is not implemented"

# Slice delete
try:
    del ints[1:3]
except NotImplementedError:
    pass
else:
    assert False, "del is not implemented"

# Slice delete
try:
    del ints[1:3]
except NotImplementedError:
    pass
else:
    assert False, "del is not implemented"

# Test that append/extend are not implemented
# append
try:
    ints.append(4)
except AttributeError:
    pass
else:
    assert False, "append is not implemented"

# extend
try:
    ints.extend([4, 5])
except AttributeError:
    pass
else:
    assert False, "extend is not implemented"

# Test set_slice nominal case
ints[1:3] = [10, 20]
assert ints[1] == 10
assert ints[2] == 20

# print(ints.tolist())

vecs = std_array.get_arr_3_vecs()
assert len(vecs) == 3
print(vecs[0])
print(vecs[1])
print(vecs[2])

# slices do not work for Eigen objects...

# v2 = vecs[:]
# assert isinstance(v2, std_array.StdVec_VectorXd)
# assert len(v2) == 3
# print(v2.tolist())
# print(v2[0])

ts = std_array.test_struct()
assert len(ts.integs) == 3
assert len(ts.vecs) == 2
print(ts.integs[:].tolist())
print(ts.vecs[0])
print(ts.vecs[1])

ts.integs[:] = 111
print("Test of set_slice for std::array<int>:", ts.integs[:].tolist())
for el in ts.integs:
    assert el == 111

ts.vecs[0][0] = 0.0
ts.vecs[1][0] = -243
print(ts.vecs[0])
print(ts.vecs[1])
