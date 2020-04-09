#!/usr/bin/env python
# -*- coding: utf-8 -*-

import distutils.dir_util
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from JHUData import *

activeWindowLo=14
activeWindowHi=21
    
timeseriesKeys,timeseries=getJHUData(False,True)  # Don't care about anything but cases now there's nothing but recovered.

def sweep(cases,window):
    return cases-np.concatenate([np.zeros((window,)),cases[:-window]])

def active(k):
    casesTotal=timeseries[k]
    casesActive=sum([sweep(casesTotal,w) for w in xrange(activeWindowLo,activeWindowHi+1)])/(1+activeWindowHi-activeWindowLo)
    return casesActive
    
for p in range(4):

    fig=plt.figure(figsize=(16,9))

    for k in timeseriesKeys:

        datestr=mdates.num2date(basedate+len(timeseries[k])-1).strftime('%Y-%m-%d')

        if p==1 and k=='Total':  # Skip global total on linear plot
            continue

        casesActive=active(k)

        if p==2 or p==3:
           casesActive=casesActive/populations[k]

        x=[t+basedate for t in range(len(casesActive))]
        plt.plot(x,casesActive,label=descriptions[k],color=colors[k],linewidth=3.0*widthScale[k])

        # TODO: Also label/annotate active case peaks?
        if p<=1 or casesActive[-1]>=1e-7:  # Don't label lines once they've dropped off the bottom
            plt.text(
                x[-1]+0.25,
                casesActive[-1],
                descriptions[k],
                horizontalalignment='left',
                verticalalignment='center',
                fontdict={'size':8,'alpha':0.8,'weight':'bold','color':colors[k]}
            )

    if p==0 or p==2:
       plt.yscale('log')
       
    if p==0 or p==1:
        plt.gca().set_ylim(bottom=1.0)
    else:
        plt.gca().set_ylim(bottom=1e-7)

    plt.xticks(rotation=75,fontsize=10)
    plt.gca().set_xlim(left=basedate,right=x[-1])
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(
        {
            0: 'Active cases ({}-{} day duration).  Log-scale.  Data to {}.',
            1: 'Active cases ({}-{} day duration).  Omits global total.  Data to {}',
            2: 'Active cases ({}-{} day duration).  Proportion of population, log-scale.  Data to {}',
            3: 'Active cases ({}-{} day duration).  Proportion of population.  Data to {}'
        }[p].format(activeWindowLo,activeWindowHi,datestr)
    )

    plt.legend(loc='upper left',fontsize='medium')

    plt.subplots_adjust(right=0.9)
    
    distutils.dir_util.mkpath('output')
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
