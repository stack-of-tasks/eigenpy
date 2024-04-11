# When running CMake for the first time set build type to release
set CMAKE_BUILD_TYPE="Release"

# Setup ccache
set CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1
