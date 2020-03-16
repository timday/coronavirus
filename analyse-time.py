#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np

from JHUData import *

timeseries=getJHUData(False)[0]

common=100.0

for k in timeseriesKeys:

    data=np.array(timeseries[k])

    if data[0]<common:
        T=np.searchsorted(data,common,'right')-1
        assert data[T]<=common
        assert common<data[T+1]
        # Want to solve common=a[T]+t*(a[T+1]-a[T]) => t=(common-a[T])/(a[T+1]-a[T])
        t=(common-data[T])/(data[T+1]-data[T])
        base=float(T)+t
    else:
        # Extrapolate the next month's data back.
        growth=(data[28]/data[0])**(1.0/28.0)
        # Solve common*growth**base = data[0] => ln(common)+ln(growth)*base=ln(data[0]).  But want sign flipped.
        base= -(math.log(data[0])-math.log(common))/math.log(growth)

    # print k,T,t,base,data[T],data[T+1]

    data[data<10.0]=np.nan

    plt.plot(np.arange(len(timeseries[k]))-base,data,color=colors[k],label=k,linewidth=3.0)

plt.grid(True)
plt.yscale('symlog')
plt.legend(loc='upper left',framealpha=0.9,fontsize='small')
             
plt.show()
