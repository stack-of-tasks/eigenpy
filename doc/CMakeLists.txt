# Doxygen configuration.
#
# Here are only the options that differ from the Doxygen defaults, or from what
# doxygen_add_docs() already presets. Defaults are kept aligned with
# jrl-cmakemodules/doxygen.cmake (v1).

# Project related configuration options
set(DOXYGEN_STRIP_FROM_PATH ${PROJECT_SOURCE_DIR})
set(DOXYGEN_STRIP_FROM_INC_PATH ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

# Build related configuration options
set(DOXYGEN_EXTRACT_ALL NO)

# Configuration options related to warning and progress messages
set(DOXYGEN_QUIET YES)
set(DOXYGEN_WARN_LOGFILE doxygen.log)

# Configuration options related to the input files
set(
  DOXYGEN_FILE_PATTERNS
  *.cpp
  *.h
  *.hpp
  *.hxx
  *.md
)
set(DOXYGEN_EXCLUDE_SYMBOLS std *::internal internal::* *::internal::*)
set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# Absolute path, so that it matches the README.md input below
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/README.md)

# Configuration options related to source browsing
set(DOXYGEN_SOURCE_BROWSER YES)

# Configuration options related to the HTML output
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION MathJax_3)
set(DOXYGEN_MATHJAX_FORMAT SVG)

# Configuration options related to the preprocessor
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)

# Configuration options related to external references
# Shipped along the HTML output, so that downstream projects can cross-reference
# the EigenPy API
set(DOXYGEN_GENERATE_TAGFILE ${CMAKE_CURRENT_BINARY_DIR}/html/${PROJECT_NAME}.doxytag)

# Configuration options related to the dot tool
set(DOXYGEN_DOT_IMAGE_FORMAT SVG)

# Configure Doxygen
doxygen_add_docs(
  doc
  ALL
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/README.md
  COMMENT "Generating API documentation with Doxygen"
)

if(INSTALL_DOCUMENTATION)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    DESTINATION share/doc/${PROJECT_NAME}
    COMPONENT doc
  )
endif()
