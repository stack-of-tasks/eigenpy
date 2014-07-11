#!/usr/bin/python

import numpy as np

'''
import libsimple

print libsimple.char()
print libsimple.str()
try:
    a = libsimple.eigenvec()
    print a
except:
    print "Error when calling simple eigenvec"

import libmystring
print libmystring.hello()
print libmystring.size("toto+5")

import libeigen
print libeigen.test()
a = np.matrix([11,2,3,4,5]).T

b = np.array([11,2,3,4,5])
#b = np.array([[15,],[1,],[1,],[1,],[1,],])
#b = np.array([ [[10,2],[3,4]],[[10,2],[3,4]] ])

print "matrix ===> "
libeigen.test2(a)
print "array ===> "
libeigen.test2(b)
'''

import libeigentemplate
# print "===> From C++ to Py"
# print libeigentemplate.test()
# print "===> From Vec C++ to Py"
# print libeigentemplate.testVec()
# print "===> From Py to C++"
a = np.random.random([5,5])
for i in range(5):
    for j in range(5):
        a[i,j] = i*5+j
#a = np.random.random([
print a
libeigentemplate.test2(a)
# print "===> From Py::slice to C++"
# b=a[1:5,1:3]
# print b
# libeigentemplate.test2(b)

# print "===> From Py::transpose to C++"
# b=a[1:5,1:3].T
# print b
# libeigentemplate.test2(b)

print "===> From py::vec to C++ Vec"
v = np.array([range(5),],np.float64).T
print v
libeigentemplate.test2Vec(v)


v = np.array(range(5),np.float64)
print "v = ", v
libeigentemplate.test2Vec(v)

v = np.array(range(10),np.float64)
v2 = v[0:10:5]
print "v2 = ", v2
libeigentemplate.test2Vec(v2)

'''
import libbnpy
a=libbnpy.array()
print a.__class__
b=libbnpy.matrix()
print b.__class__
'''
