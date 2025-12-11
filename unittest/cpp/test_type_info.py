import type_info

d = type_info.Dummy()
assert "Dummy" in d.type_info().pretty_name()

assert type_info.type_info(1).pretty_name() == "int"
assert "basic_string" in type_info.type_info("toto").pretty_name()
