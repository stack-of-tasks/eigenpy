# EigenPy 3 to EigenPy 4 {#md_doc__porting-3-to-4}

## CMake module changes

EigenPy 4 uses JRL CMake Modules 2.
This new version uses the standard CMake Boost and Python modules.
Some variables, defined by the JRL CMake Modules 1 custom Python module, are no longer available.

Here is a list of some known removed variables and their replacements:

- `PYTHON_EXT_SUFFIX`: Use `Python_SOABI` from `FindPython`

If you use JRL CMake Modules 1 in your project, you can add the following code snippet
before searching other modules:

```cmake
include("${JRL_CMAKE_MODULES}/python.cmake")
list(APPEND PYTHON_COMPONENTS Interpreter;Development.Module;NumPy)
list(REMOVE_DUPLICATES PYTHON_COMPONENTS)
FINDPYTHON()
```
