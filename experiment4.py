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
#  0: Uninfected
#  1: Initial infectious impulse
#  2: Number incubating
#  3: Number contagious 
#  4: Number observed

R=10.0

P=1e9

s=np.linspace(0.0,7.0*R,20) # Changing 20 to 40 makes too much difference... suspicious.
w=tri(s,7.0*R)
assert math.fabs(scipy.integrate.simps(w,s)-1.0) < 0.01

# Needs to return derivatives
def model(Y,t):

    y=np.array(map(lambda d: Y(t-7.0*R-d),s))

    w2=np.sum(w*y[:,2])
    w3=np.sum(w*y[:,3])

    i=(1.2/R)*(Y(t)[3]+Y(t)[1])*max(0.0,(Y(t)[0]/P))
    #i=(1.2/R)*w3+Y(t)[1]
    return np.array([
        -i,               # Uninfected            
        -Y(t)[1]/(7.0*R), # Initial impulse decays away
        i - w2,           # Infectious
        w2 - w3,          # Contagious
        0.05*Y(t)[3]      # Observed
    ])

def values_before_zero(t):
    return np.array([P,1.0,0.0,0.0,0.0])

ts=np.linspace(0.0,90.0*R,121)
ys=ddeint(model,values_before_zero,ts)

plt.plot(ts/R,ys[:,0],color='black')
plt.plot(ts/R,ys[:,1],color='purple')
plt.plot(ts/R,ys[:,2],color='blue')
plt.plot(ts/R,ys[:,3],color='green')
plt.plot(ts/R,ys[:,4],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
