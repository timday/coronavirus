#!/usr/bin/env python

import math
import numpy as np
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658,11221,14341,17187,19693,23680,27409,30553],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118,  153,  173,  183,  188,  212,  227,  265],dtype=np.float64)

assert len(china)==len(other)

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
# Implies can't infer k & j from the observations, which is interesting.

# How about time delay?  Contagious now increases with what was contagious 7 days ago.
# DSolve[{x'[t]=k*y[t],y'[t]=u*y[t-7]-v*y[t]},{x[t],y[t]},t]
# or just
# DSolve[y'[t]=u*y[t-7]-v*y[t],y[t],t]
# but neither soluble.

# How about simulation?
# Pools of fresh, incubating, infectious, immune, confirmed cases
# Variables:
#  population (initial value of fresh)
#  rate of conversion from fresh to incubating, proportional to infectious
#  rate of conversion from incubating to infectiouns
#  rate of conversion from infectious to immune
#  rate of detection (infectious cases only)
#  time at which we have infectious=1
# Simplify: don't bother with incubating?  Assume all cases confirmed?
# Just ends up with similar exponential to previous.

# Actually do day-by-day queue.  Discrete.
# State is [fresh,infectious[...queue],immune]  Make infectious queue size a fixed constant and run it for all of 1-14 days and see which fits best.

class model8c:
    def __init__(self,t0,ti,tc):
        self._t0=t0
        self._ti=ti
        self._tc=tc
        self._weight=np.array([math.sin((i+0.5)*math.pi/tc) for i in range(tc)])
        self._weight=self._weight/np.sum(self._weight)
    def __call__(self,x,t):
        n0=x[0]
        k=x[1]
        P=x[2]

        cases=0.0
        record=[]
        p=P

        pi=np.zeros(self._ti)    # Incubating.
        pc=np.zeros(self._tc)    # Contagious.
        pc[0]=n0

        tv=-self._t0
        while tv<=t[-1]:
            cases=cases+np.sum(pc)/len(pc)  # Assume detect all contagious at some point (does not remove/isolate though!)
            if tv>=t[0]:
                record.append(cases)
            i=min(p,k*(p/P)*np.sum(pc*self._weight)/len(pc))  # New incubation starts
            p=p-i                                             # Reduce uninfected population
            pc=np.insert(pc,0,pi[-1])[:-1]                    # Slide contagious cases
            pi=np.insert(pi,0,i)[:-1]                         # Slide incubating cases
            tv=tv+1

        assert len(record)==len(t)
        return record

class model9c:
    def __init__(self,t0,ti,tc,P):
        self._t0=t0
        self._ti=ti
        self._tc=tc
        self._P=P
        self._weight=np.array([math.sin((i+0.5)*math.pi/tc) for i in range(tc)])
        self._weight=self._weight/np.sum(self._weight)
    def __call__(self,x,t):
        k=x[0]

        cases=0.0
        record=[]
        p=P

        pi=np.zeros(self._ti)    # Incubating.
        pc=np.zeros(self._tc)    # Contagious.
        pc[0]=1.0

        tv=-self._t0
        while tv<=t[-1]:
            cases=cases+np.sum(pc)/len(pc)  # Assume detect all contagious at some point (does not remove/isolate though!)
            if tv>=t[0]:
                record.append(cases)
            i=min(p,k*(p/P)*np.sum(pc*self._weight)/len(pc))  # New incubation starts
            p=p-i                                             # Reduce uninfected population
            pc=np.insert(pc,0,pi[-1])[:-1]                    # Slide contagious cases
            pi=np.insert(pi,0,i)[:-1]                         # Slide incubating cases
            tv=tv+1

        assert len(record)==len(t)
        return record
 
    
def probe(data,P,where):

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

    x3s=[np.array([data[0],k,pv]) for pv in [0.000000001*P,0.00000001*P,0.0000001*P,0.000001*P,0.00001*P,0.0001*P,0.001*P,0.01*P,0.1*P,P]]
    r3s=map(lambda x3: scipy.optimize.minimize(error3,x3,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,P)]),x3s)
    r3=min(r3s,key=lambda r: r.fun)

    x4s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r4s=map(lambda x4: scipy.optimize.minimize(error4,x4,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x4s)
    r4=min(r4s,key=lambda r: r.fun)

    x5s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r5s=map(lambda x5: scipy.optimize.minimize(error5,x5,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x5s)
    r5=min(r5s,key=lambda r: r.fun)

    x6s=[np.array([data[0],k,tv]) for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r6s=map(lambda x6: scipy.optimize.minimize(error6,x6,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x6s)
    r6=min(r6s,key=lambda r: r.fun)
    
    x7s=[np.array([data[0],jkv[0],jkv[1],tv]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)] for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r7s=map(lambda x7: scipy.optimize.minimize(error7,x7,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x7s)
    r7=min(r7s,key=lambda r: r.fun)

    model8fast=True
    
    r8best=None
    model8best=None
    if where=='Other locations':
        model8T0range={False: range(1,100),True: range(12,17)}[model8fast]
    else:
        model8T0range={False: range(1,100),True: range(12,17)}[model8fast]

    model8T1range={False: range(1,28),True: range(12,17)}[model8fast]
    model8T2range={False: range(1,28),True: range(9,20)}[model8fast]
    for T0 in model8T0range:
        print 'Model8',where,T0
        for T1 in model8T1range:
            for T2 in model8T2range:
                model8=model8c(T0,T1,T2)
                    
                def error8(x):
                    return error(model8(x,days))

                x8=np.array([1.0,2.0,P])
                r8=scipy.optimize.minimize(error8,x8,method='SLSQP',options={'maxiter':10000},bounds=[(1.0,P),(0.0,np.inf),(0.0,P)])
                if r8.success and (r8best==None or r8.fun<r8best.fun):
                    r8best=r8
                    model8best=model8
    r8=r8best
    model8=model8best

    r9best=None
    model9best=None
    model9T0range=range(1,101,5)
    model9T1range=range(1,16,1)
    model9T2range=range(1,16,1)
    for T0 in model9T0range:
        print 'Model9',where,T0
        for T1 in model9T1range:
            for T2 in model9T2range:
                model9=model9c(T0,T1,T2,P)
                    
                def error9(x):
                    return error(model9(x,days))

                x9=np.array([2.0])
                r9=scipy.optimize.minimize(error9,x9,method='SLSQP',options={'maxiter':10000},bounds=[(0.0,np.inf)])
                if r9.success and (r9best==None or r9.fun<r9best.fun):
                    r9best=r9
                    model9best=model9
    r9=r9best
    model9=model9best


    print '  Model 0 score {:.6f} (success {}) {}'.format(r0.fun,r0.success,r0.x)
    print '  Model 1 score {:.6f} (success {}) {}'.format(r1.fun,r1.success,r1.x)
    print '  Model 2 score {:.6f} (success {}) {}'.format(r2.fun,r2.success,r2.x)
    print '  Model 3 score {:.6f} (success {}) {}'.format(r3.fun,r3.success,r3.x)
    print '  Model 4 score {:.6f} (success {}) {}'.format(r4.fun,r4.success,r4.x)
    print '  Model 5 score {:.6f} (success {}) {}'.format(r5.fun,r5.success,r5.x)
    print '  Model 6 score {:.6f} (success {}) {}'.format(r6.fun,r6.success,r6.x)
    print '  Model 7 score {:.6f} (success {}) {}'.format(r7.fun,r7.success,r7.x)
    print '  Model 8 score {:.6f} (success {}) {} {}'.format(r8.fun,r8.success,r8.x,[model8._t0,model8._ti,model8._tc])
    print '  Model 9 score {:.6f} (success {}) {} {}'.format(r9.fun,r9.success,r9.x,[model9._t0,model9._ti,model9._tc])

    return [r0,r1,r2,r3,r4,r5,r6,r7,r8,r9],model8,model9

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

    plt.plot(np.arange(len(data)),data,linewidth=4,color='red',label='Observed ; {} days'.format(len(data)),zorder=100)

    results,model8,model9=probe(data,P,where)
    k=map(lambda r: r.x,results)
    ok=map(lambda r: r.success,results)

    # Squash models with redundant findings
    ok[4]=ok[4] and math.fabs(k[4][1])>=0.005
    ok[5]=ok[5] and math.fabs(k[5][1])>=0.005
    ok[7]=ok[7] and math.fabs(k[7][1])>=0.005

    scores=sorted([(i,results[i].fun) for i in range(len(results)) if ok[i]],key=lambda x: x[1])
    
    def tickmarks(i):
        n=0
        if scores[0][0]==i: n=3
        elif scores[1][0]==i: n=2
        elif scores[2][0]==i: n=1
        return n*u'\u2714'
        
    t=np.arange(30+len(data))
    
    label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k[0][0],k[0][1]))+' '+tickmarks(0)
    plt.plot(t,model0(k[0],t),color='green',label=label0,zorder=1,linewidth=2)
    
    label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[1][0],k[1][1],k[1][2]))+' '+tickmarks(1)
    plt.plot(t,model1(k[1],t),color='black',label=label1,zorder=2,linewidth=2)

    label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[2][0],k[2][1],k[2][2]))+' '+tickmarks(2)
    plt.plot(t,model2(k[2],t),color='blue',label=label2,zorder=3,linewidth=2)

    if ok[3]:
        label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k[3][0],k[3][1],k[3][2]))+' '+tickmarks(3)
        plt.plot(t,model3(k[3],t),color='orange',label=label3,zorder=4,linewidth=2)

    if ok[4]:
        label4='$\\frac{dx}{dt} = (k+\\frac{j}{1+a.t}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[4][0],k[4][1],k[4][2],k[4][3]))+' '+tickmarks(4)
        plt.plot(t,model4(k[4],t),color='grey',label=label4,zorder=5,linewidth=2)
    
    if ok[5]:
        label5='$\\frac{dx}{dt} = (k+\\frac{j}{e^{a.t}}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[5][0],k[5][1],k[5][2],k[5][3]))+' '+tickmarks(5)
        plt.plot(t,model5(k[5],t),color='skyblue',label=label5,zorder=6,linewidth=2)

    if ok[6]:
        label6='$\\frac{dx}{dt} = k.(1-\\frac{t}{T}).x$ for $t \\leq T$, else $0$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, T={:.1f}$'.format(k[6][0],k[6][1],k[6][2]))+' '+tickmarks(6)
        plt.plot(t,model6(k[6],t),color='purple',label=label6,zorder=7,linewidth=2)

    if ok[7]:
        label7='$\\frac{dx}{dt} = (k+j.(1-\\frac{t}{T})).x$ for $t \\leq T$, else $k.x$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, j={:.2f}, T={:.1f}$'.format(k[7][0],k[7][1],k[7][2],k[7][3]))+' '+tickmarks(7)
        plt.plot(t,model7(k[7],t),color='pink',label=label7,zorder=8,linewidth=2)

    if ok[8]:
        label8='$S_0$'+(' ; $t_0={}, t_i={}, t_c={}, n_0={:.1f}, k={:.2f}, P={:.2g}$'.format(-model8._t0,model8._ti,model8._tc,k[8][0],k[8][1],k[8][2]))+' '+tickmarks(8)
        plt.plot(t,model8(k[8],t),color='lawngreen',label=label8,zorder=9,linewidth=2)

    if ok[9]:
        label9='$S_1$'+(' ; $t_0={}, t_i={}, t_c={}, k={:.2f}$'.format(-model9._t0,model9._ti,model9._tc,k[9][0]))+' '+tickmarks(9)
        plt.plot(t,model9(k[9],t),color='cyan',label=label9,zorder=10,linewidth=2)
        
    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xlabel('Days from 2020-01-20')
    plt.xlim(left=0.0)
    plt.legend(loc='upper left',framealpha=0.9,fontsize='x-small').set_zorder(200)
    plt.title(where+' - best fit models')

china_gain_daily=((china[1:]/china[:-1])-1.0)*100.0
other_gain_daily=((other[1:]/other[:-1])-1.0)*100.0

china_gain_weekly=(np.array([(china[i]/china[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(china))]))*100.0
other_gain_weekly=(np.array([(other[i]/other[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(other))]))*100.0

ax=plt.subplot(2,2,4)
plt.scatter(np.arange(len(china_gain_daily))+0.5,china_gain_daily,color='red',label='Mainland China (daily change)')
plt.scatter(np.arange(len(other_gain_daily))+0.5,other_gain_daily,color='blue',label='Other locations (daily change)')
plt.plot(np.arange(len(china_gain_weekly))+7.0/2.0,china_gain_weekly,color='red',label='Mainland China (1-week window)',linewidth=2)
plt.plot(np.arange(len(other_gain_weekly))+7.0/2.0,other_gain_weekly,color='blue',label='Other locations (1-week window)',linewidth=2)
plt.ylim(bottom=0.0)
plt.xlim(left=0.0)
plt.ylabel('Daily % increase rate')
plt.xlabel('Days from 2020-01-20')
plt.legend(loc='upper right',framealpha=0.9,fontsize='x-small').set_zorder(200)
plt.title('Daily % increase rate')
#plt.yscale('symlog') # Not yet

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

#plt.suptitle('Least-squares fits to confirmed cases')  # Just eats space, not very useful

plt.show()
