#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658,11221],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118,  153],dtype=np.float64)

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
# DSolve[x'[t] == k*x*(1-x/P), x[t], t]
# Wolfram Alpha doesn't solve, but is sigmoid.
# Try with sympy
#   from sympy.abc import x,t,k,P
#   from sympy import Function, dsolve, Eq, Derivative, symbols
#   x=Function('x')
#   dsolve(Eq(Derivative(x(t),t),k*x(t)*(1-x(t)/P)),x(t))
# Yields
#   x(t) == C1*exp(k*t)/(C2*exp(k*t) - 1)
# But C1/(C2-exp(-k*t)) a better representation.
# x(0)  = C1/(C2-1)
# P = x(oo) = C1/C2
# sympy.solve([Eq(x0,C1/(C2-1)),Eq(P,C1/C2)],[C1,C2])
# {C1: P*x0/(-P + x0), C2: x0/(-P + x0)}
# Mess around with signs and rearrange a bit
def model3(x,t):
    x0=x[0]
    k=x[1]
    P=x[2]
    C2=x0/(P-x0)
    C1=P*C2
    return C1/(C2+np.exp(-k*t))
# Check:
# Clearly tends to P as t->oo.
# At t=0, sympy.simplify(((P*C2)/(C2+1)).subs(C1,P*C2).subs(C2,x0/(P-x0))) is x0 as expected.

def probe(data,P):

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

    x0=np.array([data[0],0.5])
    r0=scipy.optimize.minimize(error0,x0,method='nelder-mead')
    print '  Model 0 score {:.3f}'.format(r0.fun)

    x1=np.array([data[0],0.5,0.1])
    r1=scipy.optimize.minimize(error1,x1,method='nelder-mead')
    print '  Model 1 score {:.3f}'.format(r1.fun)

    x2=np.array([data[0],0.5,0.1])
    r2=scipy.optimize.minimize(error2,x2,method='nelder-mead')
    print '  Model 2 score {:.3f}'.format(r2.fun)

    x3=np.array([data[0],0.5,P])
    r3=scipy.optimize.minimize(error3,x3,method='BFGS')  #'nelder-mead')  # BFGS and COBYLA comes up with less silly results for P
    print '  Model 3 score {:.3f} (success {})'.format(r3.fun,r3.success)

    return r0.x,r1.x,r2.x,r3.x,r3.success

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

    P={
        1: 1.4e9,
        3: 7.7e9-1.4e9,
        2: 7.7e9
        }[p]

    print '{}:'.format(where)
    
    k0,k1,k2,k3,ok3=probe(data,P)

    plt.plot(np.arange(len(data)),data,linewidth=2,color='red',label='Observed')
    
    t=np.arange(30+len(data))
    
    label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k0[0],k0[1]))
    plt.plot(t,model0(k0,t),color='green',label=label0,zorder=4)
    
    label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k1[0],k1[1],k1[2]))
    plt.plot(t,model1(k1,t),color='black',label=label1,zorder=3)
    
    label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k2[0],k2[1],k2[2]))
    plt.plot(t,model2(k2,t),color='blue',label=label2,zorder=2)

    if ok3:
        label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k3[0],k3[1],k3[2]))
        plt.plot(t,model3(k3,t),color='orange',label=label3,zorder=1)
    
    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xlabel('Days from 2020-01-20')
    plt.legend(loc='upper left',framealpha=0.9)
    plt.title(where)

plt.suptitle('Least-squares fits to confirmed cases')
plt.show()
