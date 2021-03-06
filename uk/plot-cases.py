#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import distutils.dir_util
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

extrapolationWindow=7

activeWindowLo=14
activeWindowHi=21

def sweep(cases,window):
    return cases-np.concatenate([np.zeros((window,)),cases[:-window]])

def active(casesTotal):
    casesActive=sum([sweep(casesTotal,w) for w in xrange(activeWindowLo,activeWindowHi+1)])/(1+activeWindowHi-activeWindowLo)
    return casesActive

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

for chart in [0,1,2,3,4,5]:  # 0,1 cases, 2,3 cases aligned 4,5 active cases

    fig=plt.figure(figsize=(16,9))
    
    mdayslo=None
    mdayshi=None

    texts=[]
    totalbase=0.0
    numbase=0
    
    for what in [('England',None,datetime.date(2020,3,8)),('Scotland',None,datetime.date(2020,3,8)),('Wales',None,datetime.date(2020,3,21))]:   # TODO: Northern Ireland data starts 26th March... but gappy 28th&29th?
        
        timeseries,days,codes=UKCovid19Data.getUKCovid19Data(*what,skip=set(['E06000017']))

        mdays=[mdates.date2num(d) for d in days]
        
        z=0
        for k in sorted(timeseries.keys(),key=lambda k: timeseries[k][-1],reverse=False):   # Plot highest current case counts with higher z
            cases=np.array([y for y in timeseries[k]])

            if chart==4 or chart==5:
                cases=active(cases)
            
            assert len(days)==len(cases)

            if chart==2 or chart==3:
                base=computeWhen(cases,common)
                if np.isnan(base) or base>50.0:
                    print 'Too big alignment adjustment {:.1f}, skipping'.format(base,k)
                    continue
                if what[0]=='Wales':
                    base+=14.0  # Starts 14 days after England and Scotland
            else:
                base=0.0
            
            if chart%2==1:
                cases[cases<10.0]=np.nan

            region=UKCovid19Data.whichRegion(k)

            if mdayslo==None:
                mdayslo=mdays[0]-base
            else:
                mdayslo=min(mdayslo,mdays[0]-base)
    
            if mdayshi==None:
                mdayshi=mdays[-1]-base
            else:
                mdayshi=max(mdayshi,mdays[-1]-base)

            plt.plot([d-base for d in mdays],cases,color=UKCovid19Data.colorsByRegion[region],alpha=0.8,linewidth=3.0,zorder=z)
            if not np.isnan(cases[-1]):
                totalbase+=base
                numbase+=1
                texts.append((cases[-1],codes[k],base,UKCovid19Data.colorsByRegion[region],z))  # TODO: include timeshift on chart 2.
            z+=1

    averagebase=totalbase/numbase
    
    for txt in texts:
        msg=txt[1]
        if (chart==2 or chart==3) and txt[2]!=0.0:
            msg=msg+(' ({:+.1f} days)'.format(-(txt[2]-averagebase)))
            print '{:6d} '.format(int(txt[0]))+msg
        plt.text(mdayshi+0.1,txt[0],msg,horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.8,'weight':'bold','color':txt[3]},zorder=txt[4]) 
            
    legends=[matplotlib.patches.Patch(color=UKCovid19Data.colorsByRegion[k],label=k.replace('Wales','Wales (from 2020-03-21)')) for k in sorted(UKCovid19Data.colorsByRegion.keys())]
    plt.legend(handles=legends,loc='upper left')
            
    plt.subplots_adjust(right=0.8)

    if chart==2 or chart==3:
        def relativedate(x,pos=None):
            return '{:+.0f}'.format(x-mdates.date2num(datetime.date(2020,3,7)))
        
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(relativedate))
    else:
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=75,fontsize=10)
        
    if chart==2 or chart==3:
        mdayslo+=14 # Fudge, as most of it's not interesting.  May need edit if adjust skip parameter.
    plt.xlim(left=mdayslo,right=mdayshi)
    if chart%2==1:
        plt.yscale('symlog')
        plt.ylim(bottom=10.0)
    else:
        plt.ylim(bottom=0.0)        

    if chart<=3:
        plt.gca().set_ylabel('Cumulative cases')
    else:
        plt.gca().set_ylabel('Active cases')

    if chart==0 or chart==1:
        plt.title('Cases by UTLA.  Data to {}.'.format(days[-1]))
    elif chart==2 or chart==3:
        plt.title('Cases by UTLA aligned on {} cases\nTimes +/- ahead/behind average.  Data to {}.'.format(common,days[-1]))
    else:        
        plt.title('Active cases by UTLA.  {}-{} days active duration.  Data to {}.'.format(activeWindowLo,activeWindowHi,days[-1]))

    distutils.dir_util.mkpath('output')
    plt.savefig(
        {
            0:'output/cases.png',
            1:'output/cases-log.png',
            2:'output/cases-aligned.png',
            3:'output/cases-aligned-log.png',
            4:'output/cases-active.png',
            5:'output/cases-active-log.png'
        }[chart]
        ,dpi=96
    )

plt.show()
