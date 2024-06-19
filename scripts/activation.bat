REM When running CMake for the first time set build type to release
set CMAKE_BUILD_TYPE="Release"

REM Setup ccache
set CMAKE_CXX_COMPILER_LAUNCHER=ccache

REM Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1

REM Activate color output with Ninja
set CMAKE_COLOR_DIAGNOSTICS=1
