EigenPy — Efficient Python bindings between Numpy/Eigen
======

<p align="center">
  <a href="https://opensource.org/licenses/BSD-2-Clause"><img src="https://img.shields.io/badge/License-BSD%202--Clause-green.svg" alt="License"/></a>
  <a href="https://travis-ci.org/stack-of-tasks/eigenpy"><img src="https://travis-ci.org/stack-of-tasks/eigenpy.svg?branch=master" alt="Build Status"/></a>
  <a href="https://anaconda.org/conda-forge/eigenpy"><img src="https://img.shields.io/conda/dn/conda-forge/eigenpy.svg" alt="Conda Downloads"/></a>
  <a href="https://anaconda.org/conda-forge/eigenpy"><img src="https://img.shields.io/conda/vn/conda-forge/eigenpy.svg" alt="Conda Version"/></a>
  <a href="https://badge.fury.io/py/eigenpy"><img src="https://badge.fury.io/py/eigenpy.svg" alt="PyPI version"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

**EigenPy** is an open source framework which allows to bind the famous [Eigen](http://eigen.tuxfamily.org) C++ library in Python.

**EigenPy** provides:
 - full memory sharing between Numpy and Eigen avoiding memory allocation
 - full support Eigen::Ref avoiding memory allocation
 - exposition of the Geometry module of Eigen for easy code prototyping
 - standard matrix decomposion routines of Eigen such as the Cholesky decomposition, SVD decomposition, QR decomposition, and etc.
 - full support of SWIG objects

## Setup

The installation of **EigenPy** on your computer is made easy for Linux/BSD, Mac OS X and Windows environments.

### The Conda approach

You simply need this simple line:
```
conda install eigenpy -c conda-forge
```
### Ubuntu

You can easily install **EigenPy** from binairies.

#### Add robotpkg apt repository
1. Add robotpkg as source repository to apt:
```
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
```
2. Register the authentication certificate of robotpkg:
```
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```
3. You need to run at least once apt update to fetch the package descriptions:
```
sudo apt-get update
```
#### Install EigenPy
4. The installation of **EigenPy** and its dependencies is made through the line:

For Python 2.7
```
sudo apt install robotpkg-py27-eigenpy
```
or for Python 3.{5,6,7}
```
sudo apt install robotpkg-py35-eigenpy
```
where 35 should be replaced by the python 3 you want to work this (e.g. `robotpkg-py36-eigenpy` to work with Python 3.6).

### Mac OS X

The installation of **EigenPy** on Mac OS X is made via [HomeBrew](https://brew.sh/).
You just need to register the tap of the sofware repository.

```
brew tap gepetto/homebrew-gepetto
```
and then install **EigenPy** for Python 3.x with:
```
brew install eigenpy
```

## Credits

The following people have been involved in the development of **EigenPy**:

- [Justin Carpentier](https://jcarpent.github.io) (INRIA): main developer and manager of the project
- [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard) (LAAS-CNRS): initial project instructor
- [Wolfgang Merkt](http://www.wolfgangmerkt.com/) (University of Edinburgh): ROS integration and support
- [Sean Yen](https://www.linkedin.com/in/seanyentw) (Microsoft): Windows integration
- [Loïc Estève](https://github.com/lesteve) (INRIA): Conda integration

If you have taken part to the development of **EigenPy**, feel free to add your name and contribution here.

## Acknowledgments

The development of **EigenPy** is supported by the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr) and the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr).
