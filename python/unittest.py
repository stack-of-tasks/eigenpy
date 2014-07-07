#!/usr/bin/python

import numpy as np

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
import libbnpy
a=libbnpy.array()
print a.__class__
b=libbnpy.matrix()
print b.__class__
'''
