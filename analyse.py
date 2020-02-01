#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118],dtype=np.float64)

# Straight exponential growth
# DSolve[x'[t] == k*x[t], x[t], t]
# x[t] = C*e^kt
def model0(x,t):
    C=x[0]
    k=x[1]
    return C*np.exp(k*t)

# Linearly slowing growth
# DSolve[x'[t] == (k/(1+a*t))*x[t], x[t], t]
# x[t] = C*(1+c*t)^(k/c)
def model1(x,t):
    C=x[0]
    k=x[1]
    a=x[2]
    return C*(1.0+a*t)**(k/a)

# Exponentially slowing growth
# DSolve[x'[t] == k*x[t]/exp(a*t), x[t], t]
# x[t]=C*e^(-(k/a)*e^(-at))
def model2(x,t):
    C=x[0]
    k=x[1]
    a=x[2]
    c=math.exp(-(k/a))
    return (C/c)*np.exp(-(k/a)*np.exp(-a*t))

# Logistic equation
# DSolve[x'[t] == r*x*(1-x/K), x[t], t]
# Wolfram Alpha doesn't solve, but is sigmoid
def model3(x,t):
    a=x[0]
    r=x[1]
    K=x[2]
    return K/((1.0+(1.0/a-1.0))*np.exp(-r*t))

def probe(data,K):

    def error(v):
        return np.sum((np.log(v)-np.log(data))**2)
    
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

    x1=np.array([data[0],1.0,0.1])
    r1=scipy.optimize.minimize(error1,x1,method='nelder-mead')

    x2=np.array([data[0],1.0,0.1])
    r2=scipy.optimize.minimize(error2,x2,method='nelder-mead')

    x3=np.array([data[0]/K,1.0,K])
    r3=scipy.optimize.minimize(error3,x3,method='nelder-mead')

    return r0.x,r1.x,r2.x,r3.x

for p in [1,2,3]:

    plt.subplot(2,2,p)
    
    data={
        1: china,
        3: other,
        2: china+other
    }[p]

    where={
        1: 'Mainland China',
        3: 'Other locations',
        2: 'Total'
        }[p]

    K={
        1: 1.4e9,
        3: 7.7e9-1.4e9,
        2: 7.7e9
        }[p]
    
    k0,k1,k2,k3=probe(data,K)

    plt.plot(np.arange(len(data)),data,linewidth=2,color='red',label='Observed')
    
    t=np.arange(30)
    
    label0='$\\frac{dx}{dt}=k.x$'+(' ; $x_0={:.1f}, k={:.3f}$'.format(k0[0],k0[1]))
    plt.plot(t,model0(k0,t),color='green',label=label0,zorder=3)
    
    label1='$\\frac{dx}{dt}=\\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.3f}, a={:.3f}$'.format(k1[0],k1[1],k1[2]))
    plt.plot(t,model1(k1,t),color='black',label=label1,zorder=2)
    
    label2='$\\frac{dx}{dt}=\\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.3f}, a={:.3f}$'.format(k2[0],k2[1],k2[2]))
    plt.plot(t,model2(k2,t),color='blue',label=label2,zorder=1)
    
    label3='$\\frac{dx}{dt}=r.x.(1-\\frac{x}{K})$'+(' ; $x_{{0}}={:.1f}, r={:.1f}, K={:.3g}$'.format(k3[0]*k3[1],k3[1],k3[2]))
    plt.plot(t,model3(k3,t),color='orange',label=label3,zorder=0)
    
    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xlabel('Days from 2020-01-20')
    plt.legend(loc='upper left',framealpha=0.9)
    plt.title(where)

plt.suptitle('Least-squares fits to confirmed cases')
plt.show()
