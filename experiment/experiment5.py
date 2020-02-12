#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Uninfected
#  1: Number incubating
#  2: Number contagious
#  3: Number observed

P=1e9

def impulse(t):
    if t<1.0:
        return 1.0
    else:
        return 0.0
    
# Needs to return derivatives
def model(Y,t):

    k=1.5
    
    y=Y(t)

    y7=Y(t-7.0)
    y14=Y(t-14.0)

    i_now=k*y[2]*max(0.0,y[0]/P)
    i_then=k*y7[2]*max(0.0,y7[0]/P)
    
    c_now=i_then+impulse(t)
    c_then=k*y14[2]*max(0.0,y14[0]/P)-impulse(t-7)
    
    return np.array([
        -i_now,                              # Uninfected            
        i_now-i_then*min(1.0,max(y[1],0.0)), # Incubating
        c_now-c_then*min(1.0,max(y[2],0.0)), # Recovered
        0.05*c_now                           # Observed
    ])

def values_before_zero(t):
    return np.array([P,0.0,0.0,0.0])

ts=np.linspace(0.0,120.0,121)
ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='black')
plt.plot(ts,ys[:,1],color='blue')
plt.plot(ts,ys[:,2],color='green')
plt.plot(ts,ys[:,3],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
