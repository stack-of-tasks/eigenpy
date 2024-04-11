# Remove flags setup from cxx-compiler
unset CFLAGS
unset CPPFLAGS
unset CXXFLAGS
unset DEBUG_CFLAGS
unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset LDFLAGS

# When running CMake for the first time set build type to release
export CMAKE_BUILD_TYPE="Release"

# Setup ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
