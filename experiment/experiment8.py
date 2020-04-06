#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Revisit DDE models.
# Want an active cases number: cases now - cases T weeks ago.
# Growth rate proportional to active cases.
# Maybe with herd immunity or suppression factor too but keep it simple. 

x0=10.0
k=0.2
T0=14.0  # Active window

T1=21.0   
P=1e8
a=0.1
b=0.01

def values_before_zero(t):
    v=x0*math.exp(k*t)
    return np.array([v,0.0])

# Needs to return derivatives
# Second channel is policy response
def model0(Y,t):

    ynow=Y(t)

    active=ynow[0]-Y(t-T0)[0]

    pactive1=Y(t-T1)[0]-Y(t-T1-T0)[0]
    pactive0=Y(t-T1-7.0)[0]-Y(t-T1-T0-7.0)[0]

    g=max(0.0,(pactive1/pactive0)**(1.0/7.0)-1.0)  # Growth rate in active cases over week to T1 ago
    
    dcases=(k*active*np.exp(-ynow[1]))*(1.0-ynow[0]/P)
    dpolicy=a*g-b*ynow[1]

    return np.array([dcases,dpolicy])

ts=np.linspace(0.0,210.0,2401)
ys=ddeint(model0,values_before_zero,ts)

# Considered multiple channels for various periods of incubation... but seems to get monstrously complicated quickly.

fig=plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.plot(ts,ys[:,0],color='blue')
plt.plot(ts[140:],ys[140:,0]-ys[:-140,0],color='red')
plt.yscale('symlog')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(ts,ys[:,1])
plt.grid(True)

plt.show()
