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

# Straight exponential growth
def model0(x,t):
    a=x[0]
    b=x[1]
    return a*2.0**(b*t)

# Exponential growth with reciprocal linear slowing
def model1(x,t):
    a=x[0]
    b=x[1]
    c=x[2]
    return a*2.0**(b*t/(1.0+c*t))

# Exponential growth with exponential slowing
def model2(x,t):
    a=x[0]
    b=x[1]
    c=x[2]
    return a*2.0**(b*t/(2.0**(c*t)))

# Logistic equation
def model3(x,t):
    a=x[0]
    K=x[1]
    r=x[2]
    return K/((1.0+(1.0/a-1.0))*np.exp(-r*t))

def log2(v):
    return np.log(v)/math.log(2)

def probe(data):

    def error(v):
        return np.sum((np.log2(v)-np.log2(data))**2)
    
    t=np.arange(len(data))

    def error0(x):
        return error(model0(x,t))
    def error1(x):
        return error(model1(x,t))
    def error2(x):
        return error(model2(x,t))
    def error3(x):
        return error(model3(x,t))

    x0=np.array([data[0],1.0])
    r0=scipy.optimize.minimize(error0,x0,method='nelder-mead')

    x1=np.array([data[0],1.0,0.0])
    r1=scipy.optimize.minimize(error1,x1,method='nelder-mead')

    x2=np.array([data[0],1.0,1.0])
    r2=scipy.optimize.minimize(error2,x2,method='nelder-mead')

    x3=np.array([data[0]/1e9,7.7e9,1.0])
    r3=scipy.optimize.minimize(error3,x3,method='nelder-mead')

    return r0.x,r1.x,r2.x,r3.x

k0,k1,k2,k3=probe(china+other)

plt.plot(np.arange(len(china)),china,linewidth=4,color='red',label='Observed (mainland China plus other locations)')

t=np.arange(30)

label0='$a.2^{b.t}$'+(' $a={:.1f},b={:.1f}$'.format(k0[0],k0[1]))
#plt.plot(t,k0[0]*2.0**(t*k0[1]),color='green',label=label0,zorder=3)
plt.plot(t,model0(k0,t),color='green',label=label0,zorder=3)

label1='$a.2^\\frac{b.t}{1+c.t}$'+(' $a={:.1f},b={:.1f},c={:.3f}$'.format(k1[0],k1[1],k1[2]))
plt.plot(t,model1(k1,t),color='black',label=label1,zorder=2)

label2='$a.2^\\frac{b.t}{2^{c.t}}$'+(' $a={:.1f},b={:.1f},c={:.3f}$'.format(k2[0],k2[1],k2[2]))
plt.plot(t,model2(k2,t),color='blue',label=label2,zorder=1)

label3='Sigmoid'+(' $K.x_{{0}}={:.1f},K={:.3g},r={:.1f}$'.format(k3[0]*k3[1],k3[1],k3[2]))
plt.plot(t,model3(k3,t),color='orange',label=label3,zorder=0)

plt.yscale('symlog')
plt.ylabel('Confirmed cases')
plt.xlabel('Days from 2020-01-20')
plt.legend(loc=2)
plt.title('Least-squares fits to observed data')
plt.show()
