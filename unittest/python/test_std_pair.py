from std_pair import copy, std_pair_to_tuple

t = (1, 2.0)
assert std_pair_to_tuple(t) == t
assert copy(t) == t
