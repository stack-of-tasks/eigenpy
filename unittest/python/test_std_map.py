from std_map import copy, std_map_to_dict

t = {"one": 1.0, "two": 2.0}

assert std_map_to_dict(t) == t
assert std_map_to_dict(copy(t)) == t
