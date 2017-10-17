# 
# Copyright (c) 2017 CNRS
# 
# This file is part of eigenpy
# eigenpy is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# eigenpy is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# eigenpy If not, see
# <http://www.gnu.org/licenses/>.

MACRO(SYMLINK_AND_INSTALL_HEADERS HEADERS SUBPATH)
  FOREACH(header ${HEADERS})
    GET_FILENAME_COMPONENT(headerName ${header} NAME)
    GET_FILENAME_COMPONENT(headerPath ${header} PATH)
    EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
      ${CMAKE_CURRENT_SOURCE_DIR}/${header}
      ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${SUBPATH}/${header})

    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${header}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${SUBPATH}/${headerPath}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
  ENDFOREACH(header)
ENDMACRO(SYMLINK_AND_INSTALL_HEADERS HEADERS SUBPATH)

# --- LIBRARY --- #
SET(PYWRAP ${PROJECT_NAME}_pywrap) 
MAKE_DIRECTORY("${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}")

ADD_LIBRARY(${PYWRAP} SHARED main.cpp)
TARGET_LINK_LIBRARIES(${PYWRAP} ${PROJECT_NAME}) 
TARGET_LINK_BOOST_PYTHON(${PYWRAP})
#IF(BUILD_WITH_COMMIT_VERSION)
#  TAG_LIBRARY_VERSION(${PYWRAP})
#ENDIF(BUILD_WITH_COMMIT_VERSION)
SET(${PYWRAP}_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB}/${PROJECT_NAME})

SET_PROPERTY(TARGET ${PYWRAP} PROPERTY LINKER_LANGUAGE CXX)
IF(APPLE)
  # We need to change the extension for python bindings
  SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES SUFFIX ".so")
ENDIF(APPLE)

SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}")

SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

INSTALL(TARGETS ${PYWRAP} DESTINATION ${${PYWRAP}_INSTALL_DIR})

# --- INSTALL SCRIPTS 
SET(PYTHON_FILES
  __init__.py
  )

FOREACH(python ${PYTHON_FILES})
  GET_FILENAME_COMPONENT(pythonFile ${python} NAME)
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
    ${${PROJECT_NAME}_SOURCE_DIR}/python/scripts/${python}
    ${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}/${pythonFile})
    
  # Generate pyc file
  EXECUTE_PROCESS(COMMAND
    ${PYTHON_EXECUTABLE} -m py_compile
    ${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}/${pythonFile})
  # Tag pyc file as generated.
  SET_SOURCE_FILES_PROPERTIES(
    "${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}/${pythonFile}c"
    PROPERTIES GENERATED TRUE)
    
  # Clean generated files.
  SET_PROPERTY(
    DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES
    "${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}/${pythonFile}c")
    
  INSTALL(FILES
    "${${PROJECT_NAME}_SOURCE_DIR}/python/scripts/${python}"
    "${${PROJECT_NAME}_BINARY_DIR}/python/${PROJECT_NAME}/${pythonFile}c"
    DESTINATION ${${PYWRAP}_INSTALL_DIR})
ENDFOREACH(python)

