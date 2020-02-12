#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Initial infectious impulse
#  1: Number incubating
#  2: Number contagious 
#  3: Number observed

w=np.array([min(0.5+i,6.5-i) for i in range(7)])
w=w/np.sum(w)
assert math.fabs(np.sum(w)-1.0)<0.0001

# Needs to return derivatives
def model(Y,t):

    w1=np.sum(w*np.array([Y(t-7.5-d)[1] for d in range(7)]))
    w2=np.sum(w*np.array([Y(t-7.5-d)[2] for d in range(7)]))

    return np.array([
        -Y(t)[0]/7.0,                 # Initial impulse decays away
        1.2*(Y(t)[2]+Y(t)[0]) - w1,   # Infectious
        w1 - w2,                      # Contagious
        0.05*Y(t)[2]                  # Observed
    ])

def values_before_zero(t):
    return np.array([1.0,0.0,0.0,0.0])

ts=np.arange(90)
ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='black')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='green')
plt.plot(ts,ys[:,3],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
