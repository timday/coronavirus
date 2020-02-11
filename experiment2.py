#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Number contagious
#  1: Number incubating 
#  2: Number observed

w=np.array([min(0.5+i,6.5-i) for i in range(7)])
w=w/np.sum(w)

# Needs to return derivatives
def model(Y,t):

    w0=np.sum(w*np.array([Y(t-7-d)[0] for d in range(7)]))
    ww0=np.sum(w*w*np.array([Y(t-7-d)[0] for d in range(7)]))
    w1=np.sum(w*np.array([Y(t-7-d)[1] for d in range(7)]))
    ww1=np.sum(w*w*np.array([Y(t-7-d)[1] for d in range(7)]))
    
    return np.array([
            w1 - min(1.0,Y(t)[0])*ww0,  # Huh?  Double multiply by w and w[d] in subtracted expression here.
        1.2*w0 - min(1.0,Y(t)[1])*ww1,  # Huh?  Double multiply by w and w[d] in subtracted expression here.
        0.05*w0
    ])

def values_before_zero(t):
    return np.array([1.0,0.0,0.0])

ts=np.arange(120)

ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='green')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
