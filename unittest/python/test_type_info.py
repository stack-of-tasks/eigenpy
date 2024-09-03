import type_info

d = type_info.Dummy()
assert d.type_info().pretty_name() == "Dummy"

assert type_info.type_info(1).pretty_name() == "int"
assert (
    type_info.type_info("toto").pretty_name()
    == "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>"
)
