from std_unique_ptr import make_unique_int, make_unique_v1, make_unique_null, V1

v = make_unique_int()
assert isinstance(v, int)
assert v == 10

v = make_unique_v1()
assert isinstance(v, V1)
assert v.v == 10

v = make_unique_null()
assert v is None
