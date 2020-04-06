#!/usr/bin/env python

import math
import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Revisit DDE models.
# Want an active cases number: cases now - cases T weeks ago.
# Growth rate proportional to active cases.
# Maybe with herd immunity or suppression factor too but keep it simple. 


# This is the initially infectious source
def impulse(t,d):
    if t<=0.0:
        return 0.0
    else:
        return 10.0*math.exp(-d*t)

def values_before_zero0(t):
    return np.array([0.0,0.0])

k=0.2
T=14.0
d=0.1
P=1e7
a=0.01

# Needs to return derivatives
# Need two channels 'cos something fails in ddeint otherwise
def model0(Y,t):

    ynow=Y(t)[0]
    ythen=Y(t-T)[0]

    active=ynow-ythen+impulse(t,d)

    rate=(k*active/(1.0+a*t))*(P-ynow)/P

    print t,ynow,ythen,rate
    
    return np.array([rate,0.0])

ts=np.linspace(0.0,180.0,1800)
ys=ddeint(model0,values_before_zero0,ts)[:,0]

# Considered multiple channels for various periods of incubation... but seems to get monstrously complicated quickly.

plt.plot(ts,ys,color='blue')
plt.plot(ts,ys-np.concatenate([np.zeros((14,)),ys[:-14]]),color='red')

plt.yscale('symlog')
plt.grid(True)
plt.show()
