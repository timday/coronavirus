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

    fig=plt.figure(figsize=(16,9))

    for k in timeseriesKeys:

        if p==1 and ( k=='Other' or k=='Total' ):
            continue

        casesActive=active(k)

        if p==2 or p==3:
           casesActive=casesActive/populations[k]

        plt.plot([t+basedate for t in range(len(casesActive))],casesActive,label=descriptions[k],color=colors[k],linewidth=3.0*widthScale[k])

        plt.text(
            len(casesActive)-0.5,
            casesActive[-1],
            descriptions[k],
            horizontalalignment='left',
            verticalalignment='center',
            fontdict={'size':8,'alpha':0.8,'weight':'bold','color':colors[k]}
        )

    if p==0 or p==2:
       plt.yscale('log')

    plt.xticks(rotation=75,fontsize=10)
    plt.gca().set_xlim(left=basedate)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title(
        {
            0: 'Active cases (from $\geq 30$), log-scale',
            1: 'Active cases (from $\geq 30$); omits global total',
            2: 'Active cases (from $\geq 30$), proportion of population, log-scale',
            3: 'Active cases (from $\geq 30$), proportion of population'
        }[p]
    )

    plt.legend(loc='upper left',fontsize='medium')

    plt.savefig(
        'output/'+['active-log.png','active-lin.png','active-prop-log.png','active-prop-lin.png'][p],
        dpi=96,
    )

#def on_resize(event):
#    fig.tight_layout()
#    fig.canvas.draw()
#
#fig.canvas.mpl_connect('resize_event', on_resize)

#plt.tight_layout()
plt.show()
