from std_pair import std_pair_to_tuple, copy

t = (1, 2.0)
assert std_pair_to_tuple(t) == t
assert copy(t) == t
