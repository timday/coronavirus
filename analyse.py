#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
from ddeint import ddeint
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

timeseriesKeys,timeseries=getJHUData(False,False)

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
    
    i=x[0]
    j=x[1]
    k=x[2]
    T0=x[3]
    Ti=x[4]
    Tc=x[5]

    def impulse(t):
        if t<=0.0:
            return 0.0
        else:
            return i*math.exp(-t/j)/j

    def model(Y,t):

        # Elements:
        #  0: Number incubating
        #  1: Number contagious
        #  2: Number observed
    
        y  =Y(t)        
        yp =Y(t-Ti)
        ypp=Y(t-Ti-Tc)
        ypm=Y(t-Ti-0.5*Tc)

        s=  max(0.0,1.0-y[2]/P)
        sp= max(0.0,1.0-yp[2]/P)
        spp=max(0.0,1.0-ypp[2]/P)
        spm=max(0.0,1.0-ypm[2]/P)
        
        i_now=(k*y[1]+impulse(t))*s
        i_then=(k*yp[1]+impulse(t-Ti))*sp
        
        c_now=i_then
        c_then=(k*ypp[1]+impulse(t-Ti-Tc))*spp

        c_mid=(k*ypm[1]+impulse(t-Ti-0.5*Tc))*spm
        
        return np.array([
            i_now-i_then, # Incubating
            c_now-c_then, # Contagious
            c_mid         # Observed
        ])

    def values_before_zero(t):
        return np.array([0.0,0.0,0.0])

    tms=np.concatenate([np.linspace(0.0,T0,100,endpoint=False),ts+T0])
    ys=ddeint(model,values_before_zero,tms)
    return ys[-len(ts):,2]

#print "Model8: ",model8(np.array([1.0,30,7.0,7.0]),np.linspace(0.0,30.0,31))

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

def error(v,data):
    return np.sum((np.log(v)-np.log(data))**2)

def model8error(x,days,P,data):
    #print '    Model8: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}...'.format(x[0],x[1],x[2],x[3],x[4],x[5])
    err=error(model8(x,days,P),data)
    #print '    Model8: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} : {:.3f}'.format(x[0],x[1],x[2],x[3],x[4],x[5],err)
    return  err
    
class model8minfn:
    def __init__(self,days,P,data):
        self._days=days
        self._P=P
        self._data=data
    def __call__(self,x8):
        print 'Model8 minimizing from: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(x8[0],x8[1],x8[2],x8[3],x8[4],x8[5])
        return scipy.optimize.minimize(
            lambda x: model8error(x,self._days,self._P,self._data),
            x8,
            method='SLSQP',
            options={'eps':0.5,'ftol':0.01,'maxiter':1000},
            bounds=[(0.0,100.0),(1.0,100.0),(0.0,np.inf),(1.0,1000.0),(1.0,100.0),(1.0,100.0)]   # Large (unlimited) j (&i?) causes problems?
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
    k=1.0/4.0
    a=0.1
    T=len(data)
    tolerance=0.001

    print 'Model 0'
    x0=np.array([data[0],k])
    r0=scipy.optimize.minimize(error0,x0,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf)])

    print 'Model 1'
    x1s=[np.array([data[0],k,a]),np.array([data[0],k,0.0])]
    r1s=map(lambda x1: scipy.optimize.minimize(error1,x1,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x1s)
    r1=min(r1s,key=lambda r: r.fun)

    print 'Model 2'
    x2s=[np.array([data[0],k,a]),np.array([data[0],k,0.0])]
    r2s=map(lambda x2: scipy.optimize.minimize(error2,x2,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x2s)
    r2=min(r2s,key=lambda r: r.fun)

    print 'Model 3'
    x3s=[np.array([data[0],k,pv]) for pv in [0.000000001*P,0.00000001*P,0.0000001*P,0.000001*P,0.00001*P,0.0001*P,0.001*P,0.01*P,0.1*P,P]]
    r3s=map(lambda x3: scipy.optimize.minimize(error3,x3,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,P)]),x3s)
    r3=min(r3s,key=lambda r: r.fun)

    print 'Model 4'
    x4s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r4s=map(lambda x4: scipy.optimize.minimize(error4,x4,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x4s)
    r4=min(r4s,key=lambda r: r.fun)

    print 'Model 5'
    x5s=[np.array([data[0],jkv[0],jkv[1],a]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)]]
    r5s=map(lambda x5: scipy.optimize.minimize(error5,x5,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0001,np.inf)]),x5s)
    r5=min(r5s,key=lambda r: r.fun)

    print 'Model 6'
    x6s=[np.array([data[0],k,tv]) for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r6s=map(lambda x6: scipy.optimize.minimize(error6,x6,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x6s)
    r6=min(r6s,key=lambda r: r.fun)
    
    print 'Model 7'
    x7s=[np.array([data[0],jkv[0],jkv[1],tv]) for jkv in [(k,0.0),(k/2.0,k/2.0),(0.0,k)] for tv in [0.5*T,0.75*T,T,1.5*T,2.0*T]]
    r7s=map(lambda x7: scipy.optimize.minimize(error7,x7,method='SLSQP',options={'ftol':tolerance,'maxiter':1000},bounds=[(0.0,np.inf),(0.0,np.inf),(0.0,np.inf),(0.0,np.inf)]),x7s)
    r7=min(r7s,key=lambda r: r.fun)

    if args.dde:
        print 'Model 8'
        x8s=[np.array([5.0,5.0,5.0,T0*(Ti+Tc),Ti,Tc]) for T0 in [1.0,2.0] for Ti in [14.0,21.0,28.0,35.0] for Tc in [14.0,17.5,21.0]]
        #x8s=[np.array([5.0,5.0,5.0,2.0*(28.0+15.0),28.0,15.0])]
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
    print where

    alldata=timeseries[timeseriesKeys[p]]

    data=np.array([x for x in alldata if x>=30.0])
    start=len(alldata)-len(data)
    P=populations[timeseriesKeys[p]]

    print data
    
    if p==0:
        fig1=plt.figure()
        def on_resize1(event):
            fig1.tight_layout()
            fig1.canvas.draw()
        fig1.canvas.mpl_connect('resize_event', on_resize1)

    if p==6:
        fig2=plt.figure()
        def on_resize2(event):
            fig2.tight_layout()
            fig2.canvas.draw()
        fig2.canvas.mpl_connect('resize_event', on_resize2)

    if p==12:
        fig3=plt.figure()
        def on_resize3(event):
            fig3.tight_layout()
            fig3.canvas.draw()
        fig3.canvas.mpl_connect('resize_event', on_resize3)

    if p==18:
        fig4=plt.figure()
        def on_resize4(event):
            fig4.tight_layout()
            fig4.canvas.draw()
        fig4.canvas.mpl_connect('resize_event', on_resize4)

    plt.subplot(2,3,1+(p%6))

    results=probe(data,P,where)
    k=map(lambda r: r.x,results)
    ok=map(lambda r: r.success,results)

    # Squash models with redundant findings
    ok[1]=ok[1] and math.fabs(k[1][2])>=0.005  
    ok[2]=ok[2] and math.fabs(k[2][2])>=0.005  
    ok[4]=ok[4] and math.fabs(k[4][1])>=0.005 and math.fabs(k[4][2])>=0.005 and math.fabs(k[4][3])>=0.005
    ok[5]=ok[5] and math.fabs(k[5][1])>=0.005 and math.fabs(k[5][2])>=0.005
    ok[7]=ok[7] and math.fabs(k[7][1])>=0.005 and math.fabs(k[7][2])>=0.005

    scores=sorted([(i,results[i].fun) for i in range(len(results)) if ok[i]],key=lambda x: x[1])
    
    def tickmarks(i):
        n=0
        if scores[0][0]==i: n=3
        elif scores[1][0]==i: n=2
        elif scores[2][0]==i: n=1
        return n*u'\u2714'

    def poplimit(a):
        r=np.array(a)
        r[a>P]=np.nan
        return r
    
    def date(t):
        return [basedate+x for x in t]
                
    plt.plot(date(np.arange(len(data))+start),data,linewidth=4,color='red',label='Observed ; {} days $\geq 30$ cases'.format(len(data)),zorder=100)

    alldata_nonzero=np.array(alldata)
    alldata_nonzero[alldata==0.0]=np.nan
    plt.plot(date(np.arange(len(alldata))),alldata_nonzero,linewidth=1,color='red',zorder=101)
    
    t=np.arange(30+len(data))

    if ok[0]:
        label0='$\\frac{dx}{dt} = k.x$'+(' ; $x_0={:.1f}, k={:.2f}$'.format(k[0][0],k[0][1]))+' '+tickmarks(0)
        plt.plot(date(t+start),poplimit(model0(k[0],t)),color='green',label=label0,zorder=1,linewidth=2)

    if ok[1]:
        label1='$\\frac{dx}{dt} = \\frac{k}{1+a.t}.x$'+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[1][0],k[1][1],k[1][2]))+' '+tickmarks(1)
        plt.plot(date(t+start),poplimit(model1(k[1],t)),color='black',label=label1,zorder=2,linewidth=2)

    if ok[2]:
        label2='$\\frac{dx}{dt} = \\frac{k}{e^{a.t}}.x$ '+(' ; $x_0={:.1f}, k={:.2f}, a={:.2f}$'.format(k[2][0],k[2][1],k[2][2]))+' '+tickmarks(2)
        plt.plot(date(t+start),poplimit(model2(k[2],t)),color='blue',label=label2,zorder=3,linewidth=2)

    if ok[3]:
        label3='$\\frac{dx}{dt} = k.x.(1-\\frac{x}{P})$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, P={:.2g}$'.format(k[3][0],k[3][1],k[3][2]))+' '+tickmarks(3)
        plt.plot(date(t+start),poplimit(model3(k[3],t)),color='orange',label=label3,zorder=4,linewidth=2)

    if ok[4]:
        label4='$\\frac{dx}{dt} = (k+\\frac{j}{1+a.t}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[4][0],k[4][1],k[4][2],k[4][3]))+' '+tickmarks(4)
        plt.plot(date(t+start),poplimit(model4(k[4],t)),color='grey',label=label4,zorder=5,linewidth=2)
    
    if ok[5]:
        label5='$\\frac{dx}{dt} = (k+\\frac{j}{e^{a.t}}).x$'+(' ; $x_0={:.1f}, k={:.2f}, j={:.2f}, a={:.2f}$'.format(k[5][0],k[5][1],k[5][2],k[5][3]))+' '+tickmarks(5)
        plt.plot(date(t+start),poplimit(model5(k[5],t)),color='skyblue',label=label5,zorder=6,linewidth=2)

    if ok[6]:
        label6='$\\frac{dx}{dt} = k.(1-\\frac{t}{T}).x$ for $t \\leq T$, else $0$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, T={:.1f}$'.format(k[6][0],k[6][1],k[6][2]))+' '+tickmarks(6)
        plt.plot(date(t+start),poplimit(model6(k[6],t)),color='purple',label=label6,zorder=7,linewidth=2)

    if ok[7]:
        label7='$\\frac{dx}{dt} = (k+j.(1-\\frac{t}{T})).x$ for $t \\leq T$, else $k.x$'+(' ; $x_{{0}}={:.1f}, k={:.2f}, j={:.2f}, T={:.1f}$'.format(k[7][0],k[7][1],k[7][2],k[7][3]))+' '+tickmarks(7)
        plt.plot(date(t+start),poplimit(model7(k[7],t)),color='pink',label=label7,zorder=8,linewidth=2)

    if ok[8]:
        label8='${DDE}_0$'+(' ; $i={:.1f}, j={:.1f}, k={:.1f}, T_0={:.1f}, T_i={:.1f}, T_c={:.1f}$'.format(k[8][0],k[8][1],k[8][2],-k[8][3],k[8][4],k[8][5]))+' '+tickmarks(8)
        plt.plot(date(t+start),poplimit(model8(k[8],t,P)),color='lawngreen',label=label8,zorder=9,linewidth=2)
        
    plt.yscale('symlog')
    plt.ylabel('Confirmed cases')
    plt.xticks(rotation=75,fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().set_xlim(left=basedate)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.grid(True)

    handles,labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1],labels[::-1],loc='upper left',framealpha=0.25,fontsize='xx-small').set_zorder(200)

    plt.title(where)

    if p%6==5:
        plt.tight_layout(pad=0.05)

plt.figure(figsize=(9,6))

ax=plt.subplot(1,1,1)

for p in range(len(timeseriesKeys)):

    data=timeseries[timeseriesKeys[p]]
    gain_daily=((data[1:]/data[:-1])-1.0)*100.0
    gain_weekly=(np.array([(data[i]/data[i-7])**(1.0/7.0)-1.0 for i in xrange(7,len(data))]))*100.0

    gain_daily[data[1:]<30.0]=np.nan
    gain_weekly[data[7:]<30.0]=np.nan

    day_dates=date(np.arange(len(gain_daily))+0.5)
    plt.scatter(day_dates,gain_daily,s=9.0,color=colors[timeseriesKeys[p]])
    week_dates=date(np.arange(len(gain_weekly))+7.0/2.0)
    plt.plot(week_dates,gain_weekly,color=colors[timeseriesKeys[p]],linewidth=3.0*widthScale[timeseriesKeys[p]],label=descriptions[timeseriesKeys[p]])

    plt.text(day_dates[-1]+1.5,gain_weekly[-1],descriptions[timeseriesKeys[p]],horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colors[timeseriesKeys[p]]})

for k in timeseriesKeys:
    for item in news[k]:
        txt=descriptions[k]+':'+item[1]
        date=datetime.datetime(item[0][0],item[0][1],item[0][2])

        plt.text(
            mdates.date2num(date),
            0.02,
            txt,
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=90,
            fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colors[k]}
        )
            
plt.xlim(left=basedate,right=day_dates[-1]+1)
plt.ylim(bottom=0.0)
plt.yscale('symlog')
plt.grid(True)
plt.yticks([1.0,2.0,3.0,4.0,5.0,7.5,10.0,20.0,30.0,40.0,50.0,75.0,100.0,200.0,300.0])
plt.ylabel('Daily % increase rate')
plt.xticks(rotation=75,fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_xlim(left=basedate)
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='lower left',framealpha=0.9,fontsize='medium',bbox_to_anchor=(0.01,0.01)).set_zorder(200)   # Was xx-small, but that's too small.
plt.title('Daily % increase rate and 1-week window\nStarts when >=30 cases')

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

plt.figure(figsize=(12,5))

def frequency(s):
    c=np.array([len([n for n in s if str(int(n))[0]==str(m)]) for m in range(1,10)],dtype=np.float64)
    return c/np.sum(c)

ax=plt.subplot(1,2,1)
width=0.25
plt.bar(np.arange(1,10)-width,np.log10(1.0+1.0/np.arange(1,10)),width,color='green',label='Expected')
plt.bar(np.arange(1,10)      ,frequency(timeseries['China']),width,color='red',label='Mainland China')
plt.bar(np.arange(1,10)+width,frequency(timeseries['Other']),width,color='blue',label='Other locations')
plt.legend(loc='upper right',fontsize='xx-small')
plt.ylabel('Frequency')
plt.xlabel('Leading digit')
plt.xticks(np.arange(1,10))
plt.title("Benford's Law compliance - total cases")

ax=plt.subplot(1,2,2)
width=0.25
plt.bar(np.arange(1,10)-width,np.log10(1.0+1.0/np.arange(1,10)),width,color='green',label='Expected')
plt.bar(np.arange(1,10)      ,frequency(timeseries['China'][1:]-timeseries['China'][:-1]),width,color='red',label='Mainland China')
plt.bar(np.arange(1,10)+width,frequency(timeseries['Other'][1:]-timeseries['Other'][:-1]),width,color='blue',label='Other locations')
plt.legend(loc='upper right',fontsize='xx-small')
plt.ylabel('Frequency')
plt.xlabel('Leading digit')
plt.xticks(np.arange(1,10))
plt.title("Benford's Law compliance - daily cases")

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()
