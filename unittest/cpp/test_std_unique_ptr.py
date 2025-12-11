from std_unique_ptr import (
    V1,
    UniquePtrHolder,
    make_unique_complex,
    make_unique_int,
    make_unique_null,
    make_unique_str,
    make_unique_v1,
)

v = make_unique_int()
assert isinstance(v, int)
assert v == 10

v = make_unique_v1()
assert isinstance(v, V1)
assert v.v == 10

v = make_unique_null()
assert v is None

v = make_unique_str()
assert isinstance(v, str)
assert v == "str"

v = make_unique_complex()
assert isinstance(v, complex)
assert v == 1 + 0j

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

v = unique_ptr_holder.str_ptr
assert isinstance(v, str)
assert v == "str"
# v is a copy, str_ptr will not be updated
v = "str_updated"
assert unique_ptr_holder.str_ptr == "str"

v = unique_ptr_holder.complex_ptr
assert isinstance(v, complex)
assert v == 1 + 0j
# v is a copy, complex_ptr will not be updated
v = 1 + 2j
assert unique_ptr_holder.complex_ptr == 1 + 0j
