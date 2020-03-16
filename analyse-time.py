#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from JHUData import *

timeseries=getJHUData(False)[0]

for k in timeseriesKeys:

    common=500.0

    data=np.array(timeseries[k])

    T=np.searchsorted(data,common,'right')-1
    # Want to solve common=a[T]+t*(a[T+1]-a[T]) => t=(common-a[T])/(a[T+1]-a[T])
    t=(common-data[T])/(data[T+1]-data[T])

    base=float(T)+t

    print k,T,t,base,data[T],data[T+1]

    assert data[T]<=common
    assert common<data[T+1]

    data[data<30.0]=np.nan

    plt.plot(np.arange(len(timeseries[k]))-base,data,color=colors[k],label=k,linewidth=3.0)
             
plt.yscale('symlog')
plt.legend(loc='upper left',framealpha=0.9,fontsize='small')
             
plt.show()
