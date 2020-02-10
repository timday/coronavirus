#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Elements:
#  0: Number actually contagious
#  1: Number becoming contagious = proportional to number actually contagious 7 days ago (incubation time)
#  2: Number recovering = number becoming contagious 7 days ago (contagious time)
#  3: Number observed = Proportion of becoming contagious

def model(Y,t):
    fresh=2.0*Y(t-7)[0]
    recov=Y(t-7)[1]
    return np.array([
        Y(t)[0]+fresh-recov,
        fresh,
        recov,
        Y(t)[3]+0.05*fresh
    ])

def values_before_zero(t):
    return np.array([1.0,1.0,1.0,0.0])

ts=np.arange(60)

ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys[:,0],color='green')
plt.plot(ts,ys[:,1],color='black')
plt.plot(ts,ys[:,2],color='blue')
plt.plot(ts,ys[:,3],color='red')
plt.yscale('symlog')
plt.show()
