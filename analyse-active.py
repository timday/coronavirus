#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from JHUData import *

timeseriesKeys,timeseries=getJHUData(True,True)

fig=plt.figure()

for p in range(4):

    plt.subplot(2,2,1+p)

    for k in timeseriesKeys:

        casesTotal=timeseries[0][k]
        casesRecovered=timeseries[1][k]
        casesFatal=timeseries[2][k]
    
        casesActive=casesTotal-(casesRecovered+casesFatal)
    
        casesActive[casesActive<30.0]=np.nan

        if p==2 or p==3:
           casesActive=casesActive/populations[k]

        if k=='China:Other':
            w=2.0
        else:
            w=3.0
        plt.plot(np.arange(len(casesActive)),casesActive,label=descriptions[k],color=colors[k],linewidth=w)
    
    if p==0 or p==2:
       plt.yscale('log')

    if p==2:
       plt.legend(loc='upper left',fontsize='x-small')

    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(7.0))
    plt.title(
        {
            0: 'Active cases, log-scale',
            1: 'Active cases',
            2: 'Active cases, proportion of population, log-scale',
            3: 'Active cases, proportion of population'
        }[p]
    )

plt.suptitle('Active cases ($\geq 30$)')

#def on_resize(event):
#    fig.tight_layout()
#    fig.canvas.draw()
#
#fig.canvas.mpl_connect('resize_event', on_resize)

#plt.tight_layout()
plt.show()
