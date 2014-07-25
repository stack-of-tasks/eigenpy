from se3 import *
import numpy as np
from numpy import cos,sin

fs = []
fs += [ geomLowBound,  geom1, geom2, geomRx]
fs += [kineLowBound, kine1, kine3, kine2, kine4, kineRx ]
fs += [kinegeomLowBound, kinegeom1, kinegeom3, kinegeom2, kinegeom4, kinegeomRx ]

fs = [kinegeom1, kinegeom2, kinegeom3, kinegeom4]
fs = [ kinegeom4, kinegeom2 ]

kinegeom1()

nbLoop=10

nbTest=len(fs)
t = np.zeros([nbTest,nbLoop],np.double)
kinegeom1()

for l in range(nbLoop):
    for i,f in enumerate(fs):
        t[i,l] = f()

print t
print [ np.mean(t[i,:]) for i in range(nbTest) ]

for i,f in enumerate(fs):
    print np.mean(t[i,:]),'  \t==>\t',f.__doc__.split('\n')[2]
