:: Setup ccache
set CMAKE_CXX_COMPILER_LAUNCHER=ccache

:: Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1

:: Activate color output with Ninja
set CMAKE_COLOR_DIAGNOSTICS=1

:: Set default build value only if not previously set
if not defined EIGENPY_BUILD_TYPE (set EIGENPY_BUILD_TYPE=Release)
if not defined EIGENPY_PYTHON_STUBS (set EIGENPY_PYTHON_STUBS=ON)
if not defined EIGENPY_CHOLMOD_SUPPORT (set EIGENPY_CHOLMOD_SUPPORT=OFF)
if not defined EIGENPY_ACCELERATE_SUPPORT (set EIGENPY_ACCELERATE_SUPPORT=OFF)
