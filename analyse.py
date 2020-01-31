#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658])
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118])

# No progress fit would be a.exp(b.t)
# If "learning" slows spread rate maybe a.exp((b/(1+ct)).t)

def probe(data):
    
    def model0(x):
        a=x[0]
        b=x[1]
        t=np.arange(len(data))
        return sum(((a*2.0**(b*t)-data)/data)**2)

    def model1(x):
        a=x[0]
        b=x[1]
        c=x[2]
        t=np.arange(len(data))
        return sum(((a*2.0**(b*t/(1.0+c*t))-data)/data)**2)

    def model2(x):
        a=x[0]
        b=x[1]
        c=x[2]
        t=np.arange(len(data))
        return sum(((a*2.0**(b*t/(2.0**(c*t)))-data)/data)**2)
        
    x0=np.array([data[0],1.0])
    r0=scipy.optimize.minimize(model0,x0,method='nelder-mead')

    x1=np.array([data[0],1.0,0.0])
    r1=scipy.optimize.minimize(model1,x1,method='nelder-mead')

    x2=np.array([data[0],1.0,1.0])
    r2=scipy.optimize.minimize(model2,x2,method='nelder-mead')

    return r0.x,r1.x,r2.x

k0,k1,k2=probe(china+other)

plt.plot(np.arange(len(china)),china,label='Observed (mainland China plus other locations)')

t=np.arange(30)

label0='$a.2^{b.t}$ least-squares fit'+(' $a={:.1f},b={:.1f}$'.format(k0[0],k0[1]))
plt.plot(t,k0[0]*2.0**(t*k0[1]),label=label0)

label1='$a.2^\\frac{b.t}{1+c.t}$ least-squares fit'+(' $a={:.1f},b={:.1f},c={:.3f}$'.format(k1[0],k1[1],k1[2]))
plt.plot(t,k1[0]*2.0**(t*k1[1]/(1.0+k1[2]*t)),label=label1)

label2='$a.2^\\frac{b.t}{2^{c.t}}$ least-squares fit'+(' $a={:.1f},b={:.1f},c={:.3f}$'.format(k2[0],k2[1],k2[2]))
plt.plot(t,k2[0]*2.0**(t*k2[1]/(2.0**(k2[2]*t))),label=label2)

plt.yscale('symlog')
plt.ylabel('Confirmed cases')
plt.xlabel('Days from 2020-01-20')
plt.legend(loc=2)
plt.show()
