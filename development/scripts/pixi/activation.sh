#! /bin/bash
# Activation script

if [[ $PIXI_ENVIRONMENT_PLATFORMS == *"linux"* ]];
then
  # Conda compiler is named x86_64-conda-linux-gnu-c++, ccache can't resolve it
  # (https://ccache.dev/manual/latest.html#config_compiler_type)
  export CCACHE_COMPILERTYPE=gcc
fi

# Without -isystem, some LSP can't find headers
export EIGENPY_CXX_FLAGS="$EIGENPY_CXX_FLAGS -isystem $CONDA_PREFIX/include"

# Set default build value only if not previously set
export EIGENPY_BUILD_TYPE=${EIGENPY_BUILD_TYPE:=Release}
export EIGENPY_PYTHON_STUBS=${EIGENPY_PYTHON_STUBS:=ON}
export EIGENPY_CHOLMOD_SUPPORT=${EIGENPY_CHOLMOD_SUPPORT:=OFF}
export EIGENPY_ACCELERATE_SUPPORT=${EIGENPY_ACCELERATE_SUPPORT:=OFF}
