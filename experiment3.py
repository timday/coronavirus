#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import scipy.integrate
import matplotlib.pyplot as plt

# Normalized triangle weighting t a vector
def tri(t,r):
    return np.minimum(1.0-t/r,t/r)/(0.25*r)

# Elements:
#  0: Initial infectious impulse
#  1: Number incubating
#  2: Number contagious 
#  3: Number observed

R=24.0

s=np.linspace(0.0,7.0*R,20)
w=tri(s,7.0*R)
assert math.fabs(scipy.integrate.simps(w,s)-1.0) < 0.01

# Needs to return derivatives
def model(Y,t):

    y=np.array(map(lambda d: Y(t-7.0*R-d),s))

    w1=np.sum(w*y[:,1])
    w2=np.sum(w*y[:,2])

    return np.array([
        -Y(t)[0]/(7.0*R),                      # Initial impulse decays away
        (1.2/R)*(Y(t)[2]+Y(t)[0]) - w1,  # Infectious
        w1 - w2,                               # Contagious
        0.05*Y(t)[2]                           # Observed
    ])

def values_before_zero(t):
    return np.array([1.0,0.0,0.0,0.0])

ts=np.arange(90*R)
ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='black')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='green')
plt.plot(ts,ys[:,3],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
