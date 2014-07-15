#!/usr/bin/python

import numpy as np
import libeigenpy

print "===> From MatrixXd to Py"
print libeigenpy.test()

print "===> From VectorXd to Py"
print libeigenpy.testVec()

print "===> From Py to C++"
a = np.random.random([5,5])
for i in range(5):
    for j in range(5):
        a[i,j] = i*5+j
print a
libeigenpy.test2(a)

print "===> From Py::slice to C++"
b=a[1:5,1:3]
print b
libeigenpy.test2(b)

print "===> From Py::transpose to C++"
b=a[1:5,1:3].T
libeigenpy.test2(b)

print "===> From py::vec to C++ Vec"
v = np.array([range(5),],np.float64).T
libeigenpy.test2Vec(v)

print "===> From one-dim py::vec  to C++ Vec"
v = np.array(range(5),np.float64)
libeigenpy.test2Vec(v)

print "===> From one-dim sliced py::vec  to C++ Vec"
v = np.array(range(10),np.float64)
v2 = v[0:10:3]
libeigenpy.test2Vec(v2)
