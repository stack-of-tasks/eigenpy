import numpy as np
from geometry import (
    AngleAxis,
    Quaternion,
    testInAngleAxis,
    testInQuaternion,
    testOutAngleAxis,
    testOutQuaternion,
)
from numpy import cos

verbose = True


def isapprox(a, b, epsilon=1e-6):
    if issubclass(a.__class__, np.ndarray) and issubclass(b.__class__, np.ndarray):
        return np.allclose(a, b, epsilon)
    else:
        return abs(a - b) < epsilon


# --- Quaternion ---------------------------------------------------------------
# Coefficient initialisation
verbose and print("[Quaternion] Coefficient initialisation")
q = Quaternion(1, 2, 3, 4)
q.normalize()
assert isapprox(np.linalg.norm(q.coeffs()), q.norm())
assert isapprox(np.linalg.norm(q.coeffs()), 1)

# Coefficient-vector initialisation
verbose and print("[Quaternion] Coefficient-vector initialisation")
v = np.array([0.5, -0.5, 0.5, 0.5])
for k in range(10000):
    qv = Quaternion(v)
assert isapprox(qv.coeffs(), v)

# Angle axis initialisation
verbose and print("[Quaternion] AngleAxis initialisation")
r = AngleAxis(q)
q2 = Quaternion(r)
assert q == q
assert isapprox(q.coeffs(), q2.coeffs())
assert q2.isApprox(q2)
assert q2.isApprox(q2, 1e-2)

Rq = q.matrix()
Rr = r.matrix()
assert isapprox(Rq.dot(Rq.T), np.eye(3))
assert isapprox(Rr, Rq)

# Rotation matrix initialisation
verbose and print("[Quaternion] Rotation Matrix initialisation")
qR = Quaternion(Rr)
assert q.isApprox(qR)
assert isapprox(q.coeffs(), qR.coeffs())

assert isapprox(qR[3], 1.0 / np.sqrt(30))
try:
    qR[5]
    print("Error, this message should not appear.")
except RuntimeError as e:
    if verbose:
        print("As expected, caught exception: ", e)

# --- Angle Vector ------------------------------------------------
r = AngleAxis(0.1, np.array([1, 0, 0], np.double))
if verbose:
    print("Rx(.1) = \n\n", r.matrix(), "\n")
assert isapprox(r.matrix()[2, 2], cos(r.angle))
assert isapprox(r.axis, np.array([1.0, 0, 0]))
assert isapprox(r.angle, 0.1)
assert r.isApprox(r)
assert r.isApprox(r, 1e-2)

r.axis = np.array([0, 1, 0], np.double).T
assert isapprox(r.matrix()[0, 0], cos(r.angle))

ri = r.inverse()
assert isapprox(ri.angle, -0.1)

R = r.matrix()
r2 = AngleAxis(np.dot(R, R))
assert isapprox(r2.angle, r.angle * 2)

# --- USER FUNCTIONS -----------------------------------------------------------
ro = testOutAngleAxis()
assert ro.__class__ == AngleAxis
res = testInAngleAxis(r)
assert res == r.angle

qo = testOutQuaternion()
assert qo.__class__ == Quaternion
res = testInQuaternion(q)
assert q.norm() == res
