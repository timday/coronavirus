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
    return np.array([
             np.sum(w*np.array([Y(t-7-d)[1] for d in range(7)])) - min(1.0,Y(t)[0])*np.sum(w*np.array([w[d]*Y(t-7-d)[0] for d in range(7)])),  # Huh?  Double multiply by w and w[d] in subtracted expression here.
        1.2 *np.sum(w*np.array([Y(t-7-d)[0] for d in range(7)])) - min(1.0,Y(t)[1])*np.sum(w*np.array([w[d]*Y(t-7-d)[1] for d in range(7)])),  # Huh?  Double multiply by w and w[d] in subtracted expression here.
        0.05*np.sum(w*np.array([Y(t-7-d)[1] for d in range(7)]))
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
