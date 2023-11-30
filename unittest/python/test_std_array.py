import std_array


ints = std_array.get_arr_3_ints()
print(ints[0])
print(ints[1])
print(ints[2])

_ints_slice = ints[1:2]
assert len(_ints_slice) == 2, "Slice size should be 2, got %d" % len(_ints_slice)
assert _ints_slice[0] == 1
assert _ints_slice[1] == 2

print(ints.tolist())

vecs = std_array.get_arr_3_vecs()
print(vecs[0])
print(vecs[1])
print(vecs[2])
