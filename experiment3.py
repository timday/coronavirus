#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import math
import numpy as np
from ddeint import ddeint
import scipy.integrate
import matplotlib.pyplot as plt


# Normalized triangle weighting t a vector
def tri(t,r):
    return np.minimum(1.0-t/r,t/r)/(0.25*r)

# Elements:
#  0: Number contagious
#  1: Number incubating 
#  2: Number observed

s=np.linspace(0.0,7.0,15)
w=tri(s,7.0)
print scipy.integrate.simps(w,s)

# Needs to return derivatives
def model(Y,t):

    y=np.array(map(lambda d: Y(t-7.0-d),s))

    w0=np.sum(w*y[:,0])
    w1=np.sum(w*y[:,1])
    
    return np.array([
             w1 - min(1.0,Y(t)[0])*w0,
        1.2 *w0 - min(1.0,Y(t)[1])*w1,
        0.05*w1
    ])

def values_before_zero(t):
    return np.array([1.0,0.0,0.0])

ts=np.arange(60)

ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='green')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
