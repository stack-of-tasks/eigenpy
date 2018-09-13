eigenpy
===========

Setup
-----

Before compiling this package, make sure to have initialized all git
submodules of this repo. To initialize the submodules when cloning the
repo, use:

```bash
git clone --recursive <git_url>
```

To compile this package, it is recommended to create a separate build
directory:

    mkdir _build
    cd _build
    cmake [OPTIONS] ..
    make install

Please note that CMake produces a `CMakeCache.txt` file which should
be deleted to reconfigure a package from scratch.

#### Compiling for python3 on macOS

Make sure you have boost for python3 installed. If you use homebrew, you can install it via `brew install boost-python3`.

Assuming you have python2 and python3 installed on your system, you can compile for python3 using

```
cmake -DPYTHON_EXECUTABLE=`which python3`  ..
```

In case you get an error as only the libraries for python2 are found, make sure macOS can find the python3 libraries. One way to ensure this is by adding a symbolic link to the python3 libraries like

```
ln -s /usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7.dylib /usr/loca/lib/libpython3.7.dylib
```

where the abolve file source path was determined by looking at the output of `brew ls --verbose python3 | grep libpython3`.

### Dependencies

The matrix abstract layer depends on several packages which
have to be available on your machine.

 - Libraries:
   - eigen3
 - System tools:
   - CMake (>=2.6)
   - pkg-config
   - usual compilation tools (GCC/G++, make, etc.)
 - Python 2.7
 - Boost python
