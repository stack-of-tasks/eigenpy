EigenPy
======

**EigenPy** is an open source framework which allows to bind the famous [Eigen](http://eigen.tuxfamily.org) in Python as NumPy object (as matrix or array).
**EigenPy** also exposes the Geometry module of Eigen for easy code prototyping.
**EigenPy** also supports the basic matrix decomposion routines of Eigen such as the Cholesky decomposition, SVD decomposition, QR decomposition, and etc.

## Setup

The installation of **EigenPy** on your computer is made easy for Linux/BSD and Mac OS X environments.

### Ubuntu

You can easily install **EigenPy** from binairies.

#### Add robotpkg apt repository

1. Check your distribution codename in a terminal with the following command:
```
$ lsb_release -c
Codename:       xenial
```
2. Add robotpkg as source repository to apt:
```
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub xenial robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
```
3. Register the authentication certificate of robotpkg:
```
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```
4. You need to run at least once apt update to fetch the package descriptions:
```
sudo apt-get update
```
#### Install EigenPy
5. The installation of **EigenPy** and its dependencies is made through the line:

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
and then install **EigenPy** with:
```
brew install eigenpy
```
for Python 2.7 or:
```
brew install eigenpy-python3
```
for Python 3 support.

## Credits

The following people have been involved in the development of **EigenPy**:

- [Justin Carpentier](https://jcarpent.github.io) (INRIA): main developer and manager of the project
- [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard) (LAAS-CNRS): initial project instructor
- [Wolfgang Merkt](http://www.wolfgangmerkt.com/) (University of Edinburgh): ROS integration and support
- [Sean Yen](https://www.linkedin.com/in/seanyentw) (Microsoft): Windows integration

If you have taken part to the development of **EigenPy**, feel free to add your name and contribution here.

## Acknowledgments

The development of **EigenPy** is supported by the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr) and the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr).
