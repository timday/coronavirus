#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Number contagious
#  1: Number incubating 
#  2: Number observed

w=np.array([min(0.5+i,6.5-i) for i in range(7)])
w=w/np.sum(w)
assert math.fabs(np.sum(w)-1.0)<0.0001

# Needs to return derivatives
def model(Y,t):

    i=0.0
    if 0.0<=t and t<1.0:
        i=1.0-t
    
    w0=np.sum(w*np.array([Y(t-7.5-d)[0] for d in range(7)]))
    w1=np.sum(w*np.array([Y(t-7.5-d)[1] for d in range(7)]))

    return np.array([
            w1 - w0,
        1.2*Y(t)[0] - (w1-i),
        0.05*w0
    ])

def values_before_zero(t):
    i=0.0
    if -1.0<t and t<0.0:
        i=1.0+t
    return np.array([0.0,i,0.0])

ts=np.arange(60)

ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='green')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
