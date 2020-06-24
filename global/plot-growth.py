#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import distutils.dir_util
import math
import numpy as np
import scipy.special
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from JHUData import *

def date(t):
    return [basedate+x for x in t]

timeseriesKeys,timeseries=getJHUData(False,False)
        
plt.figure(figsize=(16,9))

ax=plt.subplot(1,1,1)

for p in range(len(timeseriesKeys)):

    data=timeseries[timeseriesKeys[p]]
    gain_daily=((data[1:]/data[:-1])-1.0)*100.0
    gain_weekly=(np.array([(data[i]/data[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(data))]))*100.0

    gain_daily[data[1:]<30.0]=np.nan
    gain_weekly[data[7:]<30.0]=np.nan

    day_dates=date(np.arange(len(gain_daily))+0.5)
    plt.scatter(day_dates,gain_daily,s=9.0*widthScale[timeseriesKeys[p]],color=colors[timeseriesKeys[p]])
    week_dates=date(np.arange(len(gain_weekly))+7.0/2.0)
    plt.plot(
        week_dates,
        gain_weekly,
        color=colors[timeseriesKeys[p]],
        linewidth=3.0*widthScale[timeseriesKeys[p]],
        label=descriptions[timeseriesKeys[p]]
    )

    plt.text(day_dates[-1]+1.0,gain_weekly[-1],descriptions[timeseriesKeys[p]],horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colors[timeseriesKeys[p]]})

for k in timeseriesKeys:
    for item in news[k]:
        txt=descriptions[k]+':'+item[1]
        date=datetime.datetime(item[0][0],item[0][1],item[0][2])

        plt.text(
            mdates.date2num(date),
            0.02,
            txt,
            horizontalalignment='center',
            verticalalignment='bottom',
            rotation=90,
            fontdict={'size':8,'alpha':0.8,'weight':'bold','color':colors[k]}
        )
            
plt.xlim(left=basedate-0.5,right=day_dates[-1]+0.75)
plt.ylim(bottom=0.0)
plt.yscale('symlog')
plt.grid(True)
plt.yticks([1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,20.0,30.0,40.0,50.0,60.0,80.0,100.0,200.0,300.0])
plt.ylabel('Daily % increase rate')
plt.xticks(rotation=75,fontsize=10)
plt.yticks(fontsize=10)
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='lower left',framealpha=0.9,fontsize='small',bbox_to_anchor=(0.01,0.01)).set_zorder(200)   # Was xx-small, but that's too small.
plt.title('Daily % increase rate and 1-week window.  Data to {}.\nStarts when >=30 cases'.format(mdates.num2date(basedate+len(data)-1).strftime('%Y-%m-%d')))

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

distutils.dir_util.mkpath('output')
plt.savefig(
    'output/growth.png',
    dpi=96
)

plt.show()
