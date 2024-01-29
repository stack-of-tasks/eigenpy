from boost_variant import V1, V2, VariantHolder, make_variant

variant = make_variant()
assert isinstance(variant, V1)

v1 = V1()
v1.v = 10

v2 = V2()
v2.v = "c"

variant_holder = VariantHolder()

variant_holder.variant = v1
assert isinstance(variant_holder.variant, V1)
assert variant_holder.variant == v1.v
variant_holder.variant = 100
assert variant_holder.variant == 100
assert v1 == 100
v1 = 1000
assert variant_holder.variant == 1000
assert v1 == 1000

variant_holder.variant = v2
assert isinstance(variant_holder.variant, V1)
assert variant_holder.variant == v2.v
