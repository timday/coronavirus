#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658,11221,14341,17187,19693],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118,  153,  173,  183,  188],dtype=np.float64)

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

# What about learning models with a floor on growth rate?

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

# What about a model where the rate falls off linearly until some time T (model invalid after that point)
# DSolve[x'[t] == k*(1-t/T)*x[t],x[t],t]
# Solution C*exp(kt*(1-t/(2*T)))
def model6(x,t):
    C=x[0]
    k=x[1]
    T=x[2]
    s=np.minimum(t,T)
    return C*np.exp(k*s*(1.0-s/(2.0*T)))

# And again a variant with a constant floor?
# DSolve[x'[t] == (k+j*(1-t/T))*x[t],x[t],t]
# Solution C*exp(kt+jt*(1-t/(2*T)))
def model7(x,t):
    C=x[0]
    k=x[1]
    j=x[2]
    T=x[3]
    s=np.minimum(t,T)
    return C*np.exp(k*t+j*s*(1.0-s/(2.0*T)))  # NB Continues to evolve at rate k once j influence reaches limit

# TODO: Consider models with two variables... an infected but not yet contagious population.
# DSolve[{x'[t]=k*y[t],y'[t]=k*y[t]-j*y[t]},...) ... but that's daft, just growth at (k-j) rate.

# Try this anyway:
# DSolve[{x'[t]=k*y[t],y'[t]=j*y[t]},{x[t],y[t]},t]
# Is just a linear function of an exponential in jt anyway.

def probe(data,P):

    def error(v):
        return np.sum((np.log(v)-np.log(data))**2)
    
    days=np.arange(len(data))

    def error0(x):
        return error(model0(x,days))
    def error1(x):
        return error(model1(x,days))
    def error2(x):
        return error(model2(x,days))
    def error3(x):
        return error(model3(x,days))
    def error4(x):
        return error(model4(x,days))
    def error5(x):
        return error(model5(x,days))
    def error6(x):
        return error(model6(x,days))
    def error7(x):
        return error(model7(x,days))

    # 'nelder-mead' works good for the first three.
    # BFGS and COBYLA also seem useful/relatively stable (may not take bounds though).  SLSQP seems to be default when there are bounds.

    # Initial values to guess for k and a
    k=1.0/3.0
    a=0.1
    T=len(data)
    
    x0=np.array([data[0],k])
    r0=scipy.optimize.minimize(error0,x0,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf)])

    x1=np.array([data[0],k,a])
    r1=scipy.optimize.minimize(error1,x1,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)])

    x2=np.array([data[0],k,a])
    r2=scipy.optimize.minimize(error2,x2,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)])  

    x3=np.array([data[0],k,P])
    r3=scipy.optimize.minimize(error3,x3,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,P)])

    x4s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r4s=map(lambda x4: scipy.optimize.minimize(error4,x4,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x4s)
    r4=min(r4s,key=lambda r: r.fun)

    x5s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r5s=map(lambda x5: scipy.optimize.minimize(error5,x5,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x5s)
    r5=min(r5s,key=lambda r: r.fun)    

    x6s=[np.array([data[0],k,tv]) for tv in [0.5*T,T,2.0*T]]
    r6s=map(lambda x6: scipy.optimize.minimize(error6,x6,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x6s)
    r6=min(r6s,key=lambda r: r.fun)
    
    x7s=[np.array([data[0],jkv[0],jkv[1],tv]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)] for tv in [0.5*T,T,2.0*T]]
    r7s=map(lambda x7: scipy.optimize.minimize(error7,x7,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x7s)
    r7=min(r7s,key=lambda r: r.fun)
    
    print '  Model 0 score {:.6f} (success {}) {}'.format(r0.fun,r0.success,r0.x)
    print '  Model 1 score {:.6f} (success {}) {}'.format(r1.fun,r1.success,r1.x)
    print '  Model 2 score {:.6f} (success {}) {}'.format(r2.fun,r2.success,r2.x)
    print '  Model 3 score {:.6f} (success {}) {}'.format(r3.fun,r3.success,r3.x)
    print '  Model 4 score {:.6f} (success {}) {}'.format(r4.fun,r4.success,r4.x)
    print '  Model 5 score {:.6f} (success {}) {}'.format(r5.fun,r5.success,r5.x)
    print '  Model 6 score {:.6f} (success {}) {}'.format(r6.fun,r6.success,r6.x)
    print '  Model 7 score {:.6f} (success {}) {}'.format(r7.fun,r7.success,r7.x)

    return r0.x,r1.x,r2.x,r3.x,r3.success,r4.x,r4.success,r5.x,r5.success,r6.x,r6.success,r7.x,r7.success

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
    
    k0,k1,k2,k3,ok3,k4,ok4,k5,ok5,k6,ok6,k7,ok7=probe(data,P)

    plt.plot(np.arange(len(data)),data,linewidth=4,color='red',label='Observed',zorder=10)
    
    t=np.arange(30+len(data))
    
    label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k0[0],k0[1]))
    plt.plot(t,model0(k0,t),color='green',label=label0,zorder=9,linewidth=2)
    
    label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k1[0],k1[1],k1[2]))
    plt.plot(t,model1(k1,t),color='black',label=label1,zorder=8,linewidth=2)

    label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k2[0],k2[1],k2[2]))
    plt.plot(t,model2(k2,t),color='blue',label=label2,zorder=7,linewidth=2)

    if ok3:
        label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k3[0],k3[1],k3[2]))
        plt.plot(t,model3(k3,t),color='orange',label=label3,zorder=6,linewidth=2)

    if ok4 and math.fabs(k4[1])>0.01:
        label4='$\\frac{dx}{dt} = (k+\\frac{j}{1+a.t}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k4[0],k4[1],k4[2],k4[3]))
        plt.plot(t,model4(k4,t),color='grey',label=label4,zorder=5,linewidth=2)
    
    if ok5 and math.fabs(k5[1])>0.01:
        label5='$\\frac{dx}{dt} = (k+\\frac{j}{e^{a.t}}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k5[0],k5[1],k5[2],k5[3]))
        plt.plot(t,model5(k5,t),color='skyblue',label=label5,zorder=4,linewidth=2)

    if ok6:
        label6='$\\frac{dx}{dt} = k.(1-\\frac{t}{T}).x$ for $t \\leq T$, else $0$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, T={:.1f}$'.format(k6[0],k6[1],k6[2]))
        plt.plot(t,model6(k6,t),color='purple',label=label6,zorder=3,linewidth=2)

    if ok7 and math.fabs(k7[1])>0.01:
        label7='$\\frac{dx}{dt} = (k+j.(1-\\frac{t}{T})).x$ for $t \\leq T$, else $k.x$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, j={:.2f}, T={:.1f}$'.format(k7[0],k7[1],k7[2],k7[3]))
        plt.plot(t,model7(k7,t),color='pink',label=label7,zorder=2,linewidth=2)
        
    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xlabel('Days from 2020-01-20')
    plt.legend(loc='upper left',framealpha=0.9).set_zorder(11)
    plt.title(where+' - best fit models')

china_gain_daily=(china[1:]/china[:-1])-1.0
other_gain_daily=(other[1:]/other[:-1])-1.0

china_gain_weekly=np.array([(china[i]/china[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(china))])
other_gain_weekly=np.array([(other[i]/other[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(other))])

ax=plt.subplot(2,2,4)
plt.scatter(np.arange(len(china_gain_daily))+1.0,china_gain_daily,color='red',label='Mainland China (daily change)')
plt.scatter(np.arange(len(other_gain_daily))+1.0,other_gain_daily,color='blue',label='Other locations (daily change)')
plt.plot(np.arange(len(china_gain_weekly))+7.0/2.0,china_gain_weekly,color='red',label='Mainland China (1-week window)',linewidth=2)
plt.plot(np.arange(len(other_gain_weekly))+7.0/2.0,other_gain_weekly,color='blue',label='Other locations (1-week window)',linewidth=2)
plt.ylim(bottom=0.0)
plt.ylabel('Daily % increase rate')
plt.xlabel('Days from 2020-01-20')
plt.legend(loc='upper right',framealpha=0.9).set_zorder(11)
plt.title('Daily % increase rate')

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

#plt.suptitle('Least-squares fits to confirmed cases')  # Just eats space, not very useful

plt.show()
