#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from JHUData import *

timeseriesKeys,timeseries=getJHUData(True,True)

def active(k):

    casesTotal=timeseries[0][k]
    casesRecovered=timeseries[1][k]
    casesFatal=timeseries[2][k]
    
    casesActive=casesTotal-(casesRecovered+casesFatal)
    
    casesActive[casesActive<30.0]=np.nan

    return casesActive
    
for p in range(4):

    fig=plt.figure()

    for k in timeseriesKeys:

        casesActive=active(k)

        if p==2 or p==3:
           casesActive=casesActive/populations[k]

        plt.plot(np.arange(len(casesActive)),casesActive,label=descriptions[k],color=colors[k],linewidth=5.0*widthScale[k])

        plt.text(
            len(casesActive)-0.5,
            casesActive[-1],
            descriptions[k],
            horizontalalignment='left',
            verticalalignment='center',
            fontdict={'size':10,'alpha':0.75,'weight':'bold','color':colors[k]}
        )

    if p==0 or p==2:
       plt.yscale('log')

    plt.xlim(left=0,right=len(casesActive)-1)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(7.0))
    plt.title(
        {
            0: 'Active cases (from $\geq 30$), log-scale',
            1: 'Active cases (from $\geq 30$)',
            2: 'Active cases (from $\geq 30$), proportion of population, log-scale',
            3: 'Active cases (from $\geq 30$), proportion of population'
        }[p]
    )

    plt.legend(loc='upper left',fontsize='medium')

#def on_resize(event):
#    fig.tight_layout()
#    fig.canvas.draw()
#
#fig.canvas.mpl_connect('resize_event', on_resize)

#plt.tight_layout()
plt.show()
