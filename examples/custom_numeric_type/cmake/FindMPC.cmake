if (MPC_INCLUDES AND MPC_LIBRARIES)
  set(MPC_FIND_QUIETLY TRUE)
endif (MPC_INCLUDES AND MPC_LIBRARIES)

find_path(MPC_INCLUDES
  NAMES
  mpc.h
  PATHS
  $ENV{MPC_INC}
  ${INCLUDE_INSTALL_DIR}
)

find_library(MPC_LIBRARIES mpc PATHS $ENV{MPC_LIB} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)

# Makes sure that mpc_include and mpc_libraries are valid
# https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html
find_package_handle_standard_args(MPC DEFAULT_MSG
                                  MPC_INCLUDES MPC_LIBRARIES)
mark_as_advanced(MPC_INCLUDES MPC_LIBRARIES)