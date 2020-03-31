#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Use data from https://github.com/coviddata/covid-api 'cos JHU seem to have abandoned US states in timeseries.
# Contrary to changes mentioned in https://github.com/CSSEGISandData/COVID-19/issues/1250

parser=argparse.ArgumentParser(description='Analyse coronavirus data.')
parser.add_argument('--nodc',action='store_true',default=False,help='Exclude DC datapoint.')
args=parser.parse_args()

def value(x):
    if x=='':
        return 0
    else:
        return int(x)

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
    
stateCode={
    'Alabama'       :'AL',
    'Alaska'        :'AK',
    'Arizona'       :'AZ',
    'Arkansas'      :'AR',
    'California'    :'CA',
    'Colorado' 	    :'CO',
    'Connecticut'   :'CT',
    'Delaware' 	    :'DE',     
    'Florida' 	    :'FL',
    'Georgia' 	    :'GA',
    'Hawaii' 	    :'HI',
    'Idaho' 	    :'ID',
    'Illinois' 	    :'IL',
    'Indiana' 	    :'IN',
    'Iowa' 	    :'IA',
    'Kansas' 	    :'KS',
    'Kentucky' 	    :'KY',
    'Louisiana'     :'LA',
    'Maine' 	    :'ME',
    'Maryland' 	    :'MD',
    'Massachusetts' :'MA',
    'Michigan' 	    :'MI',
    'Minnesota'     :'MN',
    'Mississippi'   :'MS',
    'Missouri' 	    :'MO',
    'Montana' 	    :'MT',
    'Nebraska' 	    :'NE',
    'Nevada' 	    :'NV',
    'New Hampshire' :'NH',
    'New Jersey'    :'NJ',
    'New Mexico'    :'NM',
    'New York' 	    :'NY',
    'North Carolina':'NC',
    'North Dakota'  :'ND',
    'Ohio' 	    :'OH',
    'Oklahoma' 	    :'OK',
    'Oregon' 	    :'OR',
    'Pennsylvania'  :'PA',
    'Rhode Island'  :'RI',
    'South Carolina':'SC',
    'South Dakota'  :'SD',
    'Tennessee'     :'TN',
    'Texas' 	    :'TX',
    'Utah' 	    :'UT',
    'Vermont' 	    :'VT',
    'Virginia' 	    :'VA',
    'Washington'    :'WA',
    'West Virginia' :'WV',
    'Wisconsin'     :'WI',
    'Wyoming' 	    :'WY',
    'District of Columbia':'DC'
}

def getCases():
    csvfile=open('data/cases.csv')
    reader=csv.reader(csvfile)
    timeseries={}
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        if row[1]=='United States' and row[0] in stateCode:
    
            where=row[0]
    
            if not where in frozenset(['Virgin Islands','Guam','Puerto Rico']):

                where=stateCode[where]
                
                if not where in timeseries:
                    timeseries[where]=np.zeros(len(row[2:]))
    
                timeseries[where]+=np.array(map(lambda x: value(x),row[2:]),dtype=np.float64)
    
    assert len(timeseries)==51  # Should be 5

    return timeseries

def getVotes(which):
    csvfile=open('data/1976-2016-president.csv')  # Data from https://doi.org/10.7910/DVN/42MVDX/MFU99O / https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/42MVDX/MFU99O
    reader=csv.reader(csvfile)
    votes={}
    totalvotes={}
    firstRow=True    
    for row in reader:

        if firstRow:
            firstRow=False
            continue

        when=['2016','2012','2008'][which]
        who=['Trump, Donald J.','Obama, Barack H.','Obama, Barack H.'][which]
        if row[0]==when and row[6]=='US President' and row[7]==who:
            state=row[2]
            v=float(row[10])
            t=float(row[11])
            if not state in votes:
                votes[state]=v
                totalvotes[state]=t
            else:
                votes[state]+=v
                assert totalvotes[state]==t
                
    return {k:votes[k]/totalvotes[k] for k in votes.keys()},totalvotes
    
cases=getCases()

window=7

for p in range(3):

    fig=plt.figure(figsize=(8,6))
    
    votes,total=getVotes(p)
    
    rate={k:(cases[k][-1]/cases[k][-1-window])**(1.0/window)-1.0 for k in cases.keys() if cases[k][-1-window]>0.0}
    nohistory=[k for k in cases.keys() if cases[k][-1-window]==0.0]
    if (len(nohistory))>0:
        print 'Not enough history for',nohistory
    
    assert len(cases)==len(votes)

    if args.nodc:
        del rate['DC']
    
    x=np.array([100.0*votes[k] for k in rate.keys()])
    y=np.array([100.0*rate[k] for k in rate.keys()])
    w=np.array([total[k] for k in rate.keys()])
    s=np.sqrt(w/1000.0)

    plt.scatter(x,y,s=s,color='tab:blue')
    
    for i in xrange(len(x)):
        plt.text(x[i]+0.5,y[i]+0.5,rate.keys()[i])
    
    # Unweighted regression line
    r=scipy.stats.linregress(x,y)
    #print r
    gradient,intercept,r_value,p_value,std_err=r
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:orange')
    
    # Weighted regression line
    coef=np.polyfit(x,y,1,w=w)
    #print coef
    ry=coef[1]+coef[0]*rx  # Highest power first
    plt.plot(rx,ry,color='tab:red')
    rw_value=corr(x,y,w)

    what=['2016 Trump','2012 Obama','2008 Obama'][p]
    plt.xlabel('{} vote'.format(what))
    plt.ylabel('Daily % increase rate (last {} days)'.format(window))
    
    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
    vals=ax.get_xticks()
    ax.set_xticklabels(['{:,.1f}%'.format(x) for x in vals])

    plt.title('Virus cases growth rate vs. {} vote by state.\nRegression lines: weighted r={:.2f} (red), unweighted r={:.2f} (orange)'.format(what,rw_value,r_value))

    plt.savefig('output/trumpton-{}.png'.format(['2016','2012','2008'][p]),dpi=96)

plt.show()
