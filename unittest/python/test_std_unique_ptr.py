from std_unique_ptr import (
    make_unique_int,
    make_unique_v1,
    make_unique_null,
    V1,
    UniquePtrHolder,
)

v = make_unique_int()
assert isinstance(v, int)
assert v == 10

v = make_unique_v1()
assert isinstance(v, V1)
assert v.v == 10

v = make_unique_null()
assert v is None

unique_ptr_holder = UniquePtrHolder()

v = unique_ptr_holder.int_ptr
assert isinstance(v, int)
assert v == 20
# v is a copy, int_ptr will not be updated
v = 10
assert unique_ptr_holder.int_ptr == 20

v = unique_ptr_holder.v1_ptr
assert isinstance(v, V1)
assert v.v == 200
# v is a ref, v1_ptr will be updated
v.v = 10
assert unique_ptr_holder.v1_ptr.v == 10


v = unique_ptr_holder.null_ptr
assert v is None
