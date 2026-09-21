# An example of using eigenpy to extend Python using Eigen linear algebra package with custom numeric types

The custom numeric types are Boost.Multiprecision's variable-precision
`mpfr_float` and `mpc_complex`. Because an all-zero mpfr/mpc number is the
*uninitialized* state (a null limb pointer), not a valid zero, this example
specializes `eigenpy::user_type_traits<>` (see `include/header.hpp`) so that
eigenpy guards numpy's zero-filled, never-constructed buffer slots. Without
those specializations, operations on freshly created arrays
(`np.zeros((3,), dtype=MpfrFloat)`, `np.dot`, `np.full`, casts, ...) crash
inside libmpfr.

## Building

This project builds standalone against an installed eigenpy (it is not part
of eigenpy's build). It needs cmake >= 3.22, Eigen, Boost (with
Boost.Python), MPFR and MPC.

1. Make a build directory.  Move into it
2. `cmake ../`
3. `make`
4. `make install`
5. Move back up a directory
6. `python3 examplescript.py`

## Testing

`pytest test/` from this directory (with the build directory on
`PYTHONPATH`) runs the test suite, including the operations that used to
crash before eigenpy's `user_type_traits` guards.
