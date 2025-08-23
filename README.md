EigenPy — Versatile and efficient Python bindings between Numpy and Eigen
======

<p align="center">
  <a href="https://opensource.org/licenses/BSD-2-Clause"><img src="https://img.shields.io/badge/License-BSD%202--Clause-green.svg" alt="License"/></a>
  <a href="https://github.com/stack-of-tasks/eigenpy/workflows/linux.yml"><img alt="Build Status" src="https://github.com/stack-of-tasks/eigenpy/actions/workflows/linux.yml/badge.svg?branch=devel" /></a>
  <a href="https://anaconda.org/conda-forge/eigenpy"><img src="https://img.shields.io/conda/dn/conda-forge/eigenpy.svg" alt="Conda Downloads"/></a>
  <a href="https://anaconda.org/conda-forge/eigenpy"><img src="https://img.shields.io/conda/vn/conda-forge/eigenpy.svg" alt="Conda Version"/></a>
  <a href="https://badge.fury.io/py/eigenpy"><img src="https://badge.fury.io/py/eigenpy.svg" alt="PyPI version"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linter: ruff"></a>
</p>

**EigenPy** is an open-source framework that allows the binding of the famous [Eigen](http://eigen.tuxfamily.org) C++ library in Python via Boost.Python.

**EigenPy** provides:

- full memory sharing between Numpy and Eigen, avoiding memory allocation
- full support Eigen::Ref avoiding memory allocation
- full support of the Eigen::Tensor module
- exposition of the Geometry module of Eigen for easy code prototyping
- standard matrix decomposion routines of Eigen such as the Cholesky, SVD and QR decompositions
- full support of SWIG objects
- full support of runtime declaration of Numpy scalar types
- extended API to expose several STL types and some of their Boost equivalents: `optional` types, `std::pair`, maps, variants...
- full support of vectorization between C++ and Python (all the hold objects are properly aligned in memory)

## Installation

The installation of **EigenPy** on your computer is made easy for Linux/BSD, Mac OS X, and Windows environments.

### Conda

You simply need this simple line:
```bash
conda install eigenpy -c conda-forge
```

### Ubuntu

You can easily install **EigenPy** from binaries.

#### Add robotpkg apt repository

1. Register the authentication certificate of robotpkg:
```
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc
```
2. Add robotpkg as source repository to apt:
```
sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg
EOF
```
3. You need to run at least one apt update to fetch the package descriptions:
```bash
sudo apt update
```

#### Install EigenPy

4. The installation of **EigenPy** and its dependencies is made through the line:

```bash
sudo apt install robotpkg-py35-eigenpy
```
where 35 should be replaced by the Python 3, you want to work this (e.g., `robotpkg-py36-eigenpy` to work with Python 3.6).

### Mac OS X

The installation of **EigenPy** on Mac OS X is made via [HomeBrew](https://brew.sh/).
You just need to register the tap of the software repository.

```
brew tap gepetto/homebrew-gepetto
```
and then install **EigenPy** for Python 3.x with:
```
brew install eigenpy
```

### Docker

```
docker run --rm -it ghcr.io/stack-of-tasks/eigenpy:devel
```

## Build

Build instruction can be found [here](./development/build.md)

## Credits

The following people have been involved in the development of **EigenPy**:

- [Justin Carpentier](https://jcarpent.github.io) (Inria): main developer and manager of the project
- [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard) (LAAS-CNRS): initial project instructor
- [Wolfgang Merkt](http://www.wolfgangmerkt.com/) (University of Edinburgh): ROS integration and support
- [Sean Yen](https://www.linkedin.com/in/seanyentw) (Microsoft): Windows integration
- [Loïc Estève](https://github.com/lesteve) (Inria): Conda integration
- [Wilson Jallet](https://manifoldfr.github.io/) (Inria): core developer
- [Joris Vaillant](https://github.com/jorisv) (Inria): core developer and manager of the project
- [Lucas Haubert](https://www.linkedin.com/in/lucas-haubert-b668a421a/) (Inria): core developer

If you have taken part in the development of **EigenPy**, feel free to add your name and contribution here.

## Acknowledgments

The development of **EigenPy** is supported by the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr) and the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr).
