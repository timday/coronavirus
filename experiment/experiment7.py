#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

Ti=14.0
Tc=7.0

def impulse(t):
    if t<=0.0:
        return 0.0
    else:
        return 10.0*math.exp(-t)

# Elements:
#  0: Number incubating
#  1: Number contagious
#  2: Number observed

def values_before_zero0(t):
    return np.array([0.0,0.0,0.0])

# Needs to return derivatives
def model0(Y,t):

    k=1.25
    
    y=Y(t)

    print t,y

    yp=Y(t-Ti)
    ypp=Y(t-Ti-Tc)
    yppp=Y(t-Ti-0.5*Tc)

    i=impulse(t)
    ip=impulse(t-Ti)
    ipp=impulse(t-Ti-Tc)
    ippp=impulse(t-Ti-0.5*Tc)
    
    i_now=k*y[1]+i
    i_then=k*yp[1]+ip
    
    c_now=i_then
    c_then=k*ypp[1]+ipp

    m=k*yppp[1]+ippp
    
    return np.array([
        i_now-i_then, # Incubating
        c_now-c_then, # Contagious
        0.1*m         # Observed
    ])

ts=np.linspace(0.0,90.0,901)
y0s=ddeint(model0,values_before_zero0,ts)

# Considered multiple channels for various periods of incubation... but seems to get monstrously complicated quickly.

plt.plot(ts,y0s[:,0],color='blue')
plt.plot(ts,y0s[:,1],color='green')
plt.plot(ts,y0s[:,2],color='red')

plt.yscale('symlog')
plt.grid(True)
plt.show()
