#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658,11221,14341],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118,  153,  173],dtype=np.float64)

# Straight exponential growth
# DSolve[x'[t] == k*x[t], x[t], t]
# x[t] = C*e^kt
def model0(x,t):
    C=x[0]
    k=x[1]
    return C*np.exp(k*t)

# Reciprocal linearly slowing growth
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
    c=np.exp(-(k/a))
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

# TODO:
# What about a model with a floor on growth rate?

# DSolve[x'[t] == (k+j/(1+a*t))*x[t], x[t], t]
# Yes: solution C*exp((k+a*k*t+j*Log(1+a*t))/a)
def model4(x,t):
    C=x[0]
    k=x[1]
    j=x[2]
    a=x[3]
    return C*np.exp((k+a*k*t+j*np.log(1.0+a*t))/a)

# DSolve[x'[t] == (k+j/exp(a*t))*x[t], x[t], t]
# Yes: solution C*exp(k*t-j*exp(-a*t)/a)
def model5(x,t):
    C=x[0]
    k=x[1]
    j=x[2]
    a=x[3]
    c=np.exp(-(j/a))
    return (C/c)*np.exp(k*t-j*np.exp(-a*t)/a)
    
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
    def error4(x):
        return error(model4(x,t))
    def error5(x):
        return error(model5(x,t))

    # 'nelder-mead' works good for the first three.
    # BFGS and COBYLA also seem useful/relatively stable (may not take bounds though).  SLSQP seems to be default when there are bounds.
    
    x0=np.array([data[0],0.5])
    r0=scipy.optimize.minimize(error0,x0,method='SLSQP')
    print '  Model 0 score {:.3f}'.format(r0.fun)

    x1=np.array([data[0],0.5,0.1])
    r1=scipy.optimize.minimize(error1,x1,method='SLSQP')
    print '  Model 1 score {:.3f}'.format(r1.fun)

    x2=np.array([data[0],0.5,0.1])
    r2=scipy.optimize.minimize(error2,x2,method='SLSQP')  
    print '  Model 2 score {:.3f}'.format(r2.fun)

    x3=np.array([data[0],0.5,P])
    r3=scipy.optimize.minimize(error3,x3,method='SLSQP',bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,P)])
    print '  Model 3 score {:.3f} (success {})'.format(r3.fun,r3.success)

    x4=np.array([data[0],0.25,0.25,0.1])
    r4=scipy.optimize.minimize(error4,x4,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)])
    print '  Model 4 score {:.3f} (success {})'.format(r4.fun,r4.success)

    x5=np.array([data[0],0.25,0.25,0.1])
    r5=scipy.optimize.minimize(error5,x5,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)])
    print '  Model 5 score {:.3f} (success {})'.format(r5.fun,r5.success)

    return r0.x,r1.x,r2.x,r3.x,r3.success,r4.x,r4.success,r5.x,r5.success

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
    
    k0,k1,k2,k3,ok3,k4,ok4,k5,ok5=probe(data,P)

    plt.plot(np.arange(len(data)),data,linewidth=4,color='red',label='Observed')
    
    t=np.arange(30+len(data))
    
    label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k0[0],k0[1]))
    plt.plot(t,model0(k0,t),color='green',label=label0,zorder=4,linewidth=2)
    
    label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k1[0],k1[1],k1[2]))
    plt.plot(t,model1(k1,t),color='black',label=label1,zorder=3,linewidth=2)

    if ok4 and math.fabs(k4[1])>0.01:
        label4='$\\frac{dx}{dt} = (k+\\frac{j}{1+a.t}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k4[0],k4[1],k4[2],k4[3]))
        plt.plot(t,model4(k4,t),color='grey',label=label4,zorder=3,linewidth=2)
    
    label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k2[0],k2[1],k2[2]))
    plt.plot(t,model2(k2,t),color='blue',label=label2,zorder=2,linewidth=2)

    if ok5 and math.fabs(k5[1])>0.01:
        label5='$\\frac{dx}{dt} = (k+\\frac{j}{e^{a.t}}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k5[0],k5[1],k5[2],k5[3]))
        plt.plot(t,model5(k5,t),color='skyblue',label=label5,zorder=2,linewidth=2)

    if ok3:
        label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k3[0],k3[1],k3[2]))
        plt.plot(t,model3(k3,t),color='orange',label=label3,zorder=1,linewidth=2)

    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xlabel('Days from 2020-01-20')
    plt.legend(loc='upper left',framealpha=0.9)
    plt.title(where)

china_gain_daily=(china[1:]/china[:-1])-1.0
other_gain_daily=(other[1:]/other[:-1])-1.0

china_gain_weekly=np.array([(china[i]/china[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(china))])
other_gain_weekly=np.array([(other[i]/other[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(other))])

ax=plt.subplot(2,2,4)
plt.scatter(np.arange(len(china_gain_daily))+1.0,china_gain_daily,color='red',label='Mainland China (day change)')
plt.scatter(np.arange(len(other_gain_daily))+1.0,other_gain_daily,color='blue',label='Other locations (day change)')
plt.plot(np.arange(len(china_gain_weekly))+7.0,china_gain_weekly,color='red',label='Mainland China (week change)',linewidth=2)
plt.plot(np.arange(len(other_gain_weekly))+7.0,other_gain_weekly,color='blue',label='Other locations (week change)',linewidth=2)
plt.ylabel('Daily % increase')
plt.xlabel('Days from 2020-01-20')
plt.legend(loc='upper right',framealpha=0.9)
plt.title('Daily % increase')

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.suptitle('Least-squares fits to confirmed cases')
plt.show()
