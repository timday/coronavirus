#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

regions=UKCovid19Data.getUKRegions()

def whichRegion(code):
    if code[0]=='E':
        return regions[code]
    elif code[0]=='W':
        return 'Wales'
    elif code[0]=='S':
        return 'Scotland'
    else:
        assert False
        return '???'

colorsByRegion={
    'East Midlands'            :'tab:olive',
    'East of England'          :'tab:pink',
    'London'                   :'black',
    'North East'               :'tab:cyan',
    'North West'               :'tab:purple',
    'South East'               :'tab:gray',
    'South West'               :'tab:orange',
    'West Midlands'            :'tab:green',
    'Yorkshire and The Humber' :'tab:brown',
    'Wales'                    :'tab:red',
    'Scotland'                 :'tab:blue'
}

extrapolationWindow=7

# Compute shift in days to align data on common
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

common=30.0

for chart in [0,1,2]:

    fig=plt.figure(figsize=(16,9))
    
    mdayslo=None
    mdayshi=None
    
    for what in [('England',None,datetime.date(2020,3,7)),('Scotland',None,datetime.date(2020,3,7)),('Wales',None,datetime.date(2020,3,21))]:
        
        timeseries,days,codes=UKCovid19Data.getUKCovid19Data(*what)

        mdays=[mdates.date2num(d) for d in days]
        
        z=0
        for k in sorted(timeseries.keys(),key=lambda k: timeseries[k][-1],reverse=False):   # Plot highest current case counts with higher z
            cases=np.array([y for y in timeseries[k]])

            if chart==2:
                base=computeWhen(cases,common)
                if what[0]=='Wales':
                    base+=14.0
            else:
                base=0.0

            
            if chart>=1:
                cases[cases<10.0]=np.nan

            region=whichRegion(k)

            if mdayslo==None:
                mdayslo=mdays[0]-base
            else:
                mdayslo=min(mdayslo,mdays[0]-base)
    
            if mdayshi==None:
                mdayshi=mdays[-1]-base
            else:
                mdayshi=max(mdayshi,mdays[-1]-base)
            
            plt.plot([d-base for d in mdays],cases,color=colorsByRegion[region],alpha=0.75,linewidth=3.0,zorder=z)
            if not np.isnan(cases[-1]):
                plt.text(mdays[-1]+0.05,cases[-1],codes[k],horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colorsByRegion[region]},zorder=z)

            z+=1
        
    legends=[matplotlib.patches.Patch(color=colorsByRegion[k],label=k.replace('Wales','Wales (from 2020-03-21)')) for k in sorted(colorsByRegion.keys())]
    plt.legend(handles=legends)
    
    plt.legend(loc='upper left')
        
    plt.subplots_adjust(right=0.8)
    
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=75,fontsize=10)
    
    if chart!=2:
        plt.xlim(left=mdayslo,right=mdayshi)
    plt.ylim(bottom={0:0.0,1:10.0,2:10.0}[chart])
    plt.gca().set_ylabel('Cumulative cases')
    if chart>=1:
        plt.yscale('symlog')

    if chart==0:
        plt.title('Cases by UTLA')
    elif chart==1:
        plt.title('Cases by UTLA (log scale) from $\geq10$')
    else:
        plt.title('Cases by UTLA (log scale) from $\geq10$ aligned on {} cases'.format(common))
    
    plt.savefig(
        {
            0:'output/cases.png',
            1:'output/cases-log.png',
            2:'output/cases-aligned-log.png'
        }[chart]
        ,dpi=96
    )

plt.show()
