#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Some income by UTLA data from https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/datasets/regionalgrossdisposablehouseholdincomegdhibylocalauthorityintheuk
# HOWEVER, need to roll up lower to upper local authorities, probably weighted by population.


from collections import defaultdict
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

window=7

timeseries,dates,codes=UKCovid19Data.getUKCovid19Data(None,window+1,None)   # Need 8 days to get 7 growth rates.

interesting=frozenset(timeseries.keys())
codeRewrites=UKCovid19Data.getUKCodeRewrites(interesting)
    
rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

for k in sorted(rate.keys(),key=lambda k: rate[k],reverse=True):
    print k,codes[k],rate[k]

def getLowerTierPopulation():

    csvfile=open('data/income/Population-Table 1.csv','rb')
    reader=csv.reader(csvfile)
    rowCount=0

    result={}

    for row in reader:
        if rowCount<2:
            pass
        elif rowCount==2:
            assert row[1]=='LAU1 code'
            assert row[22]=='2016'
        else:
            code=row[1]
            result[code]=float(row[22])
            
        rowCount+=1

    return result
    
def getIncome(filename,column,what,lowerTierPopulation):
    csvfile=open('data/income/{}-Table 1.csv'.format(filename),'rb')
    reader=csv.reader(csvfile)
    rowCount=0

    income=defaultdict(float)
    population=defaultdict(float)
    
    for row in reader:
        if rowCount<2:
            pass
        elif rowCount==2:
            assert row[column].strip()==what
        else:
            ltcode=row[1]
            code=row[1]

            if not code in interesting and code in codeRewrites:
                code=codeRewrites[code]

            if code in interesting:
                population[code]+=lowerTierPopulation[ltcode]
                income[code]+=lowerTierPopulation[ltcode]*float(row[column])

        rowCount+=1
        
    return {k:income[k]/population[k] for k in income.keys()},population

correlation={}

def probe(filename,column,what,lowerTierPopulation):
    
    income,population=getIncome(filename,column,what,lowerTierPopulation)

    print what

    for k in rate.keys():
        if not k in income:
            print 'No income for',k,codes[k]
    
    print '  Highest'
    for k in sorted(rate.keys(),key=lambda k: income[k],reverse=True)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],income[k])
    print '  Lowest'
    for k in sorted(rate.keys(),key=lambda k: income[k],reverse=False)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],income[k])
    
    y=np.array([100.0*rate[k] for k in rate.keys()])
    x=np.array([income[k] for k in rate.keys()])
    w=np.array([population[k] for k in rate.keys()])
    s=np.sqrt(w/100.0)
    
    fig=plt.figure(figsize=(8,6))
    plt.scatter(x,y,color='tab:blue',alpha=0.5,label='UTLAs',s=s)
    r=scipy.stats.linregress(x,y)
    gradient,intercept,r_value,p_value,std_err=r
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:red',label='Linear regression')

    coef=np.polyfit(x,y,2)
    qy=coef[2]+coef[1]*rx+coef[0]*rx**2
    plt.plot(rx,qy,color='tab:green',label='Quadratic best fit')

    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

    plt.ylabel('Daily % increase rate\n({} to {})'.format(dates[0],dates[-1]))
    plt.xlabel(what)

    # plt.xscale('symlog') # Meh.

    plt.legend(loc='upper right')

    plt.title('England, Scotland and Wales UTLAs: {}\nr={:.3f}'.format(filename,r_value))

    plt.savefig('output/income-{}.png'.format(filename),dpi=96)

    correlation[what]=r_value

lowerTierPopulation=getLowerTierPopulation()
    
# probe('Gross disposable income',22,'2016',lowerTierPopulation)  # Not sure what this is.

probe('GDHI per head',22,'2016',lowerTierPopulation)

print

print 'Correlations, highest to lowest'
for k in sorted(correlation.keys(),key=lambda k: correlation[k],reverse=True):
    print '  {:40s}: {:.3f}'.format(k.replace('- Average score',''),correlation[k]);

plt.show()
