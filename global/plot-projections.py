#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
from ddeint import ddeint
import distutils.dir_util
import math
import numpy as np
import scipy.optimize
import scipy.special
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from multiprocessing import Pool

from JHUData import *

parser=argparse.ArgumentParser(description='Analyse coronavirus data.')
parser.add_argument('--dde',action='store_true',default=False,help='Include slow DDE modelling.')
args=parser.parse_args()

activeWindowLo=14
activeWindowHi=21
    
def sweep(cases,window):
    shifted=np.concatenate([np.zeros((window,)),cases])
    return cases-shifted[:len(cases)]

def active(casesTotal):
    casesActive=sum([sweep(casesTotal,w) for w in xrange(activeWindowLo,activeWindowHi+1)])/(1+activeWindowHi-activeWindowLo)
    return casesActive

timeseriesKeys,timeseries=getJHUData(False,False)

# Return a decent sized figure with tight redrawing on resize
def tightfig():
    fig=plt.figure(figsize=(16,9))
    def on_resize(event):
        fig.tight_layout(pad=0.1)
        fig.canvas.draw()
    fig.canvas.mpl_connect('resize_event',on_resize)
    return fig

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

def model8(x,ts,P):
    
    x0=x[0]
    k=x[1]
    a=x[2]
    T=x[3]
    
    def model(Y,t):

        ynow=Y(t)[0]
        ythen=Y(t-T)[0]

        active=ynow-ythen

        rate=(k*active/(1.0+a*t))*(P-ynow)/P

        return np.array([rate,0.0])

    def values_before_zero(t):
        if t<-T:
            return np.array([0.0,0.0])
        elif t<0.0:
            return np.array([x0*(T+t)/T,0.0])
        else:
            return np.array([x0,0.0])

    ys=ddeint(model,values_before_zero,ts)
    return ys[:,0]

def error(v,data):
    return np.sum((np.log(v)-np.log(data))**2)

def model8error(x,days,P,data):
    err=error(model8(x,days,P),data)
    return  err
    
class model8minfn:
    def __init__(self,days,P,data):
        self._days=days
        self._P=P
        self._data=data
    def __call__(self,x8):
        print 'Model8 minimizing from: {:.1f} {:.1f} {:.1f} {:.1f}'.format(x8[0],x8[1],x8[2],x8[3])
        return scipy.optimize.minimize(
            lambda x: model8error(x,self._days,self._P,self._data),
            x8,
            method='SLSQP',
            options={'eps':0.01,'ftol':0.001,'maxiter':10000},
            bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(1.0,np.inf)]   # Large (unlimited) j (&i?) causes problems?
        )
    
def probe(data,P,where):

    days=np.arange(len(data))

    def error0(x):
        return error(model0(x,days),data)
    def error1(x):
        return error(model1(x,days),data)
    def error2(x):
        return error(model2(x,days),data)
    def error3(x):
        return error(model3(x,days),data)
    def error4(x):
        return error(model4(x,days),data)
    def error5(x):
        return error(model5(x,days),data)
    def error6(x):
        return error(model6(x,days),data)
    def error7(x):
        return error(model7(x,days),data)

    # 'nelder-mead' works good for the first three.
    # BFGS and COBYLA also seem useful/relatively stable (may not take bounds though).  SLSQP seems to be default when there are bounds.

    # Initial values to guess for k and a
    k=0.25
    a=0.05
    T=len(data)
    tolerance=0.001

    print 'Model 0'
    x0=np.array([data[0],k])
    r0=scipy.optimize.minimize(error0,x0,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf)])

    print 'Model 1'
    x1s=[np.array([data[0],k,a]),np.array([data[0],k,0.0])]
    r1s=map(lambda x1: scipy.optimize.minimize(error1,x1,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x1s)
    r1=min(r1s,key=lambda r: r.fun)

    print 'Model 2'
    x2s=[
        np.array([data[0],    k,    a]),
        np.array([data[0],    k,0.2*a]),
        np.array([data[0],    k,5.0*a]),
        np.array([data[0],0.2*k,    a]),
        np.array([data[0],0.2*k,0.2*a]),
        np.array([data[0],0.2*k,5.0*a]),
        np.array([data[0],5.0*k,    a]),
        np.array([data[0],5.0*k,0.2*a]),
        np.array([data[0],5.0*k,5.0*a])
    ]
    r2s=map(lambda x2: scipy.optimize.minimize(error2,x2,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x2s)
    r2=min(r2s,key=lambda r: r.fun)

    print 'Model 3'
    x3s=[np.array([data[0],k,pv]) for pv in [0.000000001*P,0.00000001*P,0.0000001*P,0.000001*P,0.00001*P,0.0001*P,0.001*P,0.01*P,0.1*P,P]]
    r3s=map(lambda x3: scipy.optimize.minimize(error3,x3,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,P)]),x3s)
    r3=min(r3s,key=lambda r: r.fun)

    print 'Model 4'
    x4s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r4s=map(lambda x4: scipy.optimize.minimize(error4,x4,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x4s)
    r4=min(r4s,key=lambda r: r.fun)

    print 'Model 5'
    x5s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r5s=map(lambda x5: scipy.optimize.minimize(error5,x5,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x5s)
    r5=min(r5s,key=lambda r: r.fun)

    print 'Model 6'
    x6s=[np.array([data[0],k,tv]) for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r6s=map(lambda x6: scipy.optimize.minimize(error6,x6,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x6s)
    r6=min(r6s,key=lambda r: r.fun)
    
    print 'Model 7'
    x7s=[np.array([data[0],jkv[0],jkv[1],tv]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)] for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r7s=map(lambda x7: scipy.optimize.minimize(error7,x7,method='SLSQP',options={'eps':0.01,'ftol':tolerance,'maxiter':10000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x7s)
    r7=min(r7s,key=lambda r: r.fun)

    if args.dde:
        print 'Model 8'
        x8s=[np.array([data[0],k,a,T]) for k in [0.1,0.2,0.3] for a in [0.0,0.1] for T in [7.0,14.0,21.0,28.0]]
        minfn=model8minfn(days,P,data)
        pool=Pool(8)
        r8s=pool.map(minfn,x8s)
        r8=min(r8s,key=lambda r: r.fun)
    else:
        r8=r7
        r8.success=False

    print '  Model 0 score {:.6f} (success {}) {}'.format(r0.fun,r0.success,r0.x)
    print '  Model 1 score {:.6f} (success {}) {}'.format(r1.fun,r1.success,r1.x)
    print '  Model 2 score {:.6f} (success {}) {}'.format(r2.fun,r2.success,r2.x)
    print '  Model 3 score {:.6f} (success {}) {}'.format(r3.fun,r3.success,r3.x)
    print '  Model 4 score {:.6f} (success {}) {}'.format(r4.fun,r4.success,r4.x)
    print '  Model 5 score {:.6f} (success {}) {}'.format(r5.fun,r5.success,r5.x)
    print '  Model 6 score {:.6f} (success {}) {}'.format(r6.fun,r6.success,r6.x)
    print '  Model 7 score {:.6f} (success {}) {}'.format(r7.fun,r7.success,r7.x)
    print '  Model 8 score {:.6f} (success {}) {}'.format(r8.fun,r8.success,r8.x)

    return [r0,r1,r2,r3,r4,r5,r6,r7,r8]

for p in range(len(timeseriesKeys)):

    print
    print '********************'
    where=descriptions[timeseriesKeys[p]]
    print 'Projection',p,':',where

    alldata=timeseries[timeseriesKeys[p]]

    data=np.array([x for x in alldata if x>=30.0])
    start=len(alldata)-len(data)
    P=populations[timeseriesKeys[p]]

    print 'Data:',data
    
    if p%3==0:
        fig=tightfig()

    results=probe(data,P,where)
    k=map(lambda r: r.x,results)
    ok=map(lambda r: r.success,results)

    # Squash models with redundant findings
    ok[1]=ok[1] and math.fabs(k[1][2])>=0.005  
    ok[2]=ok[2] and math.fabs(k[2][2])>=0.005  
    ok[4]=ok[4] and math.fabs(k[4][1])>=0.005 and math.fabs(k[4][2])>=0.005 and math.fabs(k[4][3])>=0.005
    ok[5]=ok[5] and math.fabs(k[5][1])>=0.005 and math.fabs(k[5][2])>=0.005
    ok[7]=ok[7] and math.fabs(k[7][1])>=0.005 and math.fabs(k[7][2])>=0.005

    # List of successful model numbers, with scores, sorted best (lowest) score first
    scores=sorted([(i,results[i].fun) for i in range(len(results)) if ok[i]],key=lambda x: x[1])

    for chart in range(2):
    
        plt.subplot(2,3,1+(p%3)+3*chart)

        def tickmarks(i):
            n=0
            if len(scores)>=1 and scores[0][0]==i: n=3
            elif len(scores)>=2 and scores[1][0]==i: n=2
            elif len(scores)>=3 and scores[2][0]==i: n=1
            return n*u'\u2714'

        def priority(i):
            for j in range(len(scores)):
                if scores[j][0]==i:
                    return len(scores)-j
            return 0
        
        def munge(a):
            r=np.array(a)
            r[a>P]=np.nan

            if chart==1:  # Convert to active cases
                r=active(r)
                
            return r
        
        def date(t):
            return [basedate+x for x in t]

        plt.plot(date(np.arange(len(data))+start),munge(data),linewidth=4,color='red',label='Observed ; {} days $\geq 30$ cases'.format(len(data)),zorder=100)
    
        alldata_nonzero=np.array(alldata)
        alldata_nonzero[alldata==0.0]=np.nan
        if chart==0:
            plt.plot(date(np.arange(len(alldata))),alldata_nonzero,linewidth=1,color='red',zorder=101)
                    
        t=np.arange(60+len(data))

        # For the active cases chart, just plot the top 3 models.
        if chart==1:
            for i in range(9):
                if tickmarks(i)=='':
                    ok[i]=False
    
        if ok[0]:
            label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k[0][0],k[0][1]))+' '+tickmarks(0)
            plt.plot(date(t+start),munge(model0(k[0],t)),color='green',label=label0,zorder=priority(0),linewidth=2)
    
        if ok[1]:
            label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[1][0],k[1][1],k[1][2]))+' '+tickmarks(1)
            plt.plot(date(t+start),munge(model1(k[1],t)),color='black',label=label1,zorder=priority(1),linewidth=2)
    
        if ok[2]:
            label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[2][0],k[2][1],k[2][2]))+' '+tickmarks(2)
            plt.plot(date(t+start),munge(model2(k[2],t)),color='blue',label=label2,zorder=priority(2),linewidth=2)
    
        if ok[3]:
            label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k[3][0],k[3][1],k[3][2]))+' '+tickmarks(3)
            plt.plot(date(t+start),munge(model3(k[3],t)),color='orange',label=label3,zorder=priority(3),linewidth=2)
    
        if ok[4]:
            label4='$\\frac{dx}{dt} = (k+\\frac{j}{1+a.t}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[4][0],k[4][1],k[4][2],k[4][3]))+' '+tickmarks(4)
            plt.plot(date(t+start),munge(model4(k[4],t)),color='grey',label=label4,zorder=priority(4),linewidth=2)
        
        if ok[5]:
            label5='$\\frac{dx}{dt} = (k+\\frac{j}{e^{a.t}}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[5][0],k[5][1],k[5][2],k[5][3]))+' '+tickmarks(5)
            plt.plot(date(t+start),munge(model5(k[5],t)),color='skyblue',label=label5,zorder=priority(5),linewidth=2)
    
        if ok[6]:
            label6='$\\frac{dx}{dt} = k.(1-\\frac{t}{T}).x$ for $t \\leq T$, else $0$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, T={:.1f}$'.format(k[6][0],k[6][1],k[6][2]))+' '+tickmarks(6)
            plt.plot(date(t+start),munge(model6(k[6],t)),color='purple',label=label6,zorder=priority(6),linewidth=2)
    
        if ok[7]:
            label7='$\\frac{dx}{dt} = (k+j.(1-\\frac{t}{T})).x$ for $t \\leq T$, else $k.x$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, j={:.2f}, T={:.1f}$'.format(k[7][0],k[7][1],k[7][2],k[7][3]))+' '+tickmarks(7)
            plt.plot(date(t+start),munge(model7(k[7],t)),color='pink',label=label7,zorder=priority(7),linewidth=2)
    
        if ok[8]:
            label8='${DDE}$'+(' ; $x_0={:.1f}, k={:.1f}, a={:.1f}, T={:.1f}$'.format(k[8][0],k[8][1],k[8][2],k[8][3]))+' '+tickmarks(8)
            plt.plot(date(t+start),munge(model8(k[8],t,P)),color='lawngreen',label=label8,zorder=priority(8),linewidth=2)
            
        plt.yscale('symlog')
        plt.xticks(rotation=75,fontsize=6)
        plt.yticks(fontsize=6)
        plt.gca().set_xlim(left=basedate)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.grid(True)
    
        handles,labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1],labels[::-1],loc='upper left',framealpha=0.25,fontsize='x-small').set_zorder(200)

        if chart==0:
            plt.title(where+': cumulative cases')
            plt.ylabel('Confirmed cases')
        else:
            plt.title(where+': active cases (uniform 14-21 day duration)')
            plt.ylabel('Active cases')
        
    if p%3==2 or p==len(timeseriesKeys)-1:
        plt.tight_layout(pad=0.1)

    if p%3==2 or p==len(timeseriesKeys)-1:
        distutils.dir_util.mkpath('output')
        plt.savefig(
            'output/projections-{}.png'.format(p/3),
            dpi=96
        )
        
plt.show()
