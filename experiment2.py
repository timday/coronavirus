#!/usr/bin/env python

# DDE Delay differential equation interesting.
# But really want to integrate over a range of previous t
# which would be more like https://stackoverflow.com/a/52260839/24283

import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

def model(Y,t):
    return 2.0*Y(t-7)-0.5*Y(t)

def values_before_zero(t):
    return 1.0

ts=np.arange(60)

ys=ddeint(model,values_before_zero,ts)

plt.plot(ts,ys)
plt.yscale('symlog')
plt.show()
