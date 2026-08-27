from std_map import X, copy, copy_boost, copy_X, std_map_to_dict

t = {"one": 1.0, "two": 2.0}
t2 = {"one": 1, "two": 2, "three": 3}

assert std_map_to_dict(t) == t
assert std_map_to_dict(copy(t)) == t
m = copy_boost(t2)
assert m.todict() == t2

xmap_cpp = copy_X({"one": X(1), "two": X(2)})
print(xmap_cpp.todict())
x1 = xmap_cpp["one"]
x1.val = 11
print(xmap_cpp.todict())
assert xmap_cpp["one"].val == 11
assert xmap_cpp["two"].val == 2
