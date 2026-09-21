import numpy as np
import user_type_uninitialized as m

# GuardedScalar mimics pointer-backed scalars (e.g. Boost.Multiprecision
# mpfr/mpc) whose all-zero byte pattern is *not* a valid value: numpy
# zero-fills fresh buffers (NPY_NEEDS_INIT) but never runs a C++ constructor,
# so any operation reading such a slot would crash for those types. Instead
# of crashing, GuardedScalar records the bad read in a violation counter.
# eigenpy::user_type_traits<GuardedScalar> opts into eigenpy's guards, so
# every operation below must complete without a single bad read.

dtype = m.GuardedScalar
rows, cols = 3, 4


def assert_no_violation(context):
    count = m.get_violation_count()
    assert count == 0, f"{count} read(s) of never-constructed slots during: {context}"


def values(mat):
    return np.array([float(x) for x in mat.reshape(mat.size)])


m.reset_violation_count()

# --- creation -------------------------------------------------------------

zeros = np.zeros((rows, cols), dtype=dtype)
assert_no_violation("np.zeros")

empty = np.empty((rows, cols), dtype=dtype)
assert_no_violation("np.empty")

# getitem on never-written slots (heals in place)
assert float(zeros[0, 0]) == 0.0
assert float(empty[0, 0]) == 0.0
assert_no_violation("getitem on never-written slots")

# setitem/getitem roundtrip through np.empty
empty2 = np.empty((rows, cols), dtype=dtype)
empty2[0, 0] = dtype(42.0)
assert float(empty2[0, 0]) == 42.0
assert_no_violation("setitem/getitem roundtrip")

# repr of an unwritten array (getitem on every slot)
repr(np.empty((rows, cols), dtype=dtype))
assert_no_violation("repr of unwritten array")

# --- binary arithmetic ----------------------------------------------------

zsum = np.zeros((rows, cols), dtype=dtype) + np.zeros((rows, cols), dtype=dtype)
assert (values(zsum) == 0.0).all()
assert_no_violation("addition of unwritten zeros")

written = np.array(np.full((rows, cols), 2.0), dtype=dtype)
mixed = written + np.zeros((rows, cols), dtype=dtype)
assert (values(mixed) == 2.0).all()
assert_no_violation("addition mixing written and unwritten")

for op in (np.subtract, np.multiply):
    op(np.zeros((rows, cols), dtype=dtype), written)
    assert_no_violation(f"{op.__name__} on unwritten zeros")

# --- comparisons ----------------------------------------------------------

a = np.zeros((rows, cols), dtype=dtype)
b = np.zeros((rows, cols), dtype=dtype)
assert (a == b).all()
assert not (a != b).any()
assert (a <= b).all()
assert (a >= b).all()
assert not (a < b).any()
assert_no_violation("comparisons on unwritten zeros")

# --- unary ops ------------------------------------------------------------

neg = -np.zeros((rows, cols), dtype=dtype)
assert (values(neg) == 0.0).all()
np.square(np.zeros((rows, cols), dtype=dtype))
assert_no_violation("unary ops on unwritten zeros")

# --- matmul and dot -------------------------------------------------------

sq = np.zeros((rows, rows), dtype=dtype)
prod = sq @ sq
assert (values(prod) == 0.0).all()
assert_no_violation("matmul of unwritten zeros")

wsq = np.array(np.eye(rows), dtype=dtype)
prod = wsq @ sq
assert (values(prod) == 0.0).all()
assert_no_violation("matmul mixing written and unwritten")

v0 = np.zeros(rows, dtype=dtype)
v1 = np.zeros(rows, dtype=dtype)
assert float(np.dot(v0, v1)) == 0.0
assert float(np.inner(v0, v1)) == 0.0
assert_no_violation("np.dot/np.inner of unwritten zeros")

w0 = np.array(np.arange(1.0, rows + 1.0), dtype=dtype)
assert float(np.dot(w0, w0)) == float(
    np.dot(np.arange(1.0, rows + 1.0), np.arange(1.0, rows + 1.0))
)
assert_no_violation("np.dot of written values matches float dot")

# partial write then dot
part = np.zeros(rows, dtype=dtype)
part[0] = dtype(3.0)
assert float(np.dot(part, part)) == 9.0
assert_no_violation("np.dot of partially written vector")

# --- copy / astype (copyswap and cast loops) ------------------------------

c = np.zeros((rows, cols), dtype=dtype).copy()
assert (values(c) == 0.0).all()
assert_no_violation("copy of unwritten zeros")

as_float = np.zeros((rows, cols), dtype=dtype).astype(np.float64)
assert (as_float == 0.0).all()
assert_no_violation("astype(float64) of unwritten zeros")

from_int = np.zeros((rows, cols), dtype=np.int64).astype(dtype)
assert (values(from_int) == 0.0).all()
assert_no_violation("astype from int64")

# non-contiguous copy (strided copyswapn)
nc = np.zeros((rows, 2 * cols), dtype=dtype)[:, ::2].copy()
assert (values(nc) == 0.0).all()
assert_no_violation("non-contiguous copy")

# --- fill paths -----------------------------------------------------------

full = np.full((rows, cols), dtype(7.0))
assert (values(full) == 7.0).all()
assert_no_violation("np.full (fillwithscalar)")

# broadcast scalar into an unwritten slice
target = np.empty((rows, cols), dtype=dtype)
target[1, :] = dtype(5.0)
assert float(target[1, 2]) == 5.0
assert_no_violation("scalar broadcast into unwritten slice")

# --- truthiness / nonzero -------------------------------------------------

assert np.count_nonzero(np.zeros((rows, cols), dtype=dtype)) == 0
nz = np.zeros(rows, dtype=dtype)
nz[1] = dtype(1.0)
assert np.count_nonzero(nz) == 1
assert bool(nz[1])
assert not bool(nz[0])
assert_no_violation("truthiness / count_nonzero")

print("all user_type_uninitialized checks passed, violations: 0")
