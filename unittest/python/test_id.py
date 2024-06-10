import eigenpy

ldlt1 = eigenpy.LDLT()
ldlt2 = eigenpy.LDLT()

id1 = ldlt1.id()
id2 = ldlt2.id()

assert id1 != id2
assert id1 == ldlt1.id()
assert id2 == ldlt2.id()
