#!/usr/bin/env python
# -*- coding: utf-8 -*-

import distutils.dir_util
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from JHUData import *

timeseriesKeys,timeseriesAll=getJHUData(True,True)

extrapolationWindow=7

def computeWhen(data,common):
    if data[-1]<common:
        #Extrapolate the last month's data forward.
        growth=(data[-1]/data[-1-extrapolationWindow])**(1.0/extrapolationWindow)
        # Solve common = data[-1]*growth**base => ln(common) = ln(data[-1]+ln(growth)*base
        if growth==1.0:
            base=0.0
        else:
            base= len(data)+(math.log(common)-math.log(data[-1]))/math.log(growth)
    elif data[0]<common:
        T=np.searchsorted(data,common,'right')-1
        assert data[T]<=common
        if common==data[T]:
            t=0.0
        else:
            assert common<data[T+1]
            # Want to solve common=a[T]+t*(a[T+1]-a[T]) => t=(common-a[T])/(a[T+1]-a[T])
            t=(common-data[T])/(data[T+1]-data[T])
        base=float(T)+t
    else:
        # Extrapolate the next month's data back.
        growth=(data[extrapolationWindow]/data[0])**(1.0/extrapolationWindow)
        # Solve common*growth**base = data[0] => ln(common)+ln(growth)*base=ln(data[0]).  But want sign flipped.
        base= -(math.log(data[0])-math.log(common))/math.log(growth)
    return base

for chart in range(2):

    fig=plt.figure(figsize=(16,9))
    
    timeseries=timeseriesAll[chart]
    common={0:1000.0,1:30.0}[chart]
    ignore={0:30.0,1:10.0}[chart]

    what={0:'Total confirmed cases',1:'Total deaths'}[chart]

    whenItaly=computeWhen(timeseries['Italy'],common)
    datestr=mdates.num2date(basedate+len(timeseries['Italy'])-1).strftime('%Y-%m-%d')
    
    for k in timeseriesKeys:
    
        data=np.array(timeseries[k])
    
        base=computeWhen(data,common)
        
        # print k,T,t,base,data[T],data[T+1]
    
        data[data<ignore]=np.nan
    
        txt=descriptions[k]
        if k!='Italy':
            txt=txt+': {:+.1f} days'.format(whenItaly-base)

        dates=np.arange(len(timeseries[k]))-base

        use=[it for it in zip(dates,data) if not np.isnan(it[1])]

        if len(use)==0:
            continue
        
        dates,data=zip(*use)
        
        plt.plot(dates,data,color=colors[k],label=txt,linewidth=3.0*widthScale[k])
    
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(7.0))
    plt.grid(True)
    plt.yscale('symlog')

    plt.ylim(bottom=ignore)
    
    if chart==0:
        plt.legend(loc='lower left',framealpha=0.9,fontsize='small')
    else:
        plt.legend(loc='lower right',framealpha=0.9,fontsize='small')
        
    plt.title('{} aligned on {:d}.  Times +/- ahead/behind Italy.  Data to {}'.format(what,int(common),datestr))

    distutils.dir_util.mkpath('output')
    plt.savefig(
        'output/'+['aligned-cases.png','aligned-deaths.png'][chart],
        dpi=96,
    )

plt.show()
