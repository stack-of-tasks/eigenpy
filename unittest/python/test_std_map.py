from std_map import copy, copy_boost, std_map_to_dict

t = {"one": 1.0, "two": 2.0}
t2 = {"one": 1, "two": 2, "three": 3}

assert std_map_to_dict(t) == t
assert std_map_to_dict(copy(t)) == t
m = copy_boost(t2)
assert m.todict() == t2
