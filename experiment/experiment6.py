#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Number incubating
#  1: Number contagious
#  2: Number observed

Ti=14.0
Tc=7.0

def impulse(t):
    if t<=0.0:
        return 0.0
    else:
        return 10.0*math.exp(-t)

# Needs to return derivatives
def model(Y,t):

    k=1.25
    
    y=Y(t)

    print t,y

    yp=Y(t-Ti)
    ypp=Y(t-Ti-Tc)
    yppp=Y(t-Ti-0.5*Tc)

    i_now=k*y[1]+impulse(t)
    i_then=k*yp[1]+impulse(t-Ti)
    
    c_now=i_then
    c_then=k*ypp[1]+impulse(t-Ti-Tc)

    m=k*yppp[1]+impulse(t-Ti-0.5*Tc)
    
    return np.array([
        i_now-i_then, # Incubating
        c_now-c_then, # Contagious
        0.1*m         # Observed
    ])

def values_before_zero(t):
    return np.array([0.0,0.0,0.0])

ts=np.linspace(0.0,90.0,901)
ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='blue')
plt.plot(ts,ys[:,1],color='green')
plt.plot(ts,ys[:,2],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
