:: Set default build value only if not previously set
if not defined EIGENPY_BUILD_TYPE (set EIGENPY_BUILD_TYPE=Release)
if not defined EIGENPY_PYTHON_STUBS (set EIGENPY_PYTHON_STUBS=ON)
if not defined EIGENPY_CHOLMOD_SUPPORT (set EIGENPY_CHOLMOD_SUPPORT=OFF)
if not defined EIGENPY_ACCELERATE_SUPPORT (set EIGENPY_ACCELERATE_SUPPORT=OFF)
