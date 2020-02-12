#!/usr/bin/env python

import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Normalized triangle weighting t a vector
def tri(t,r):
    return np.minimum(1.0-t/r,t/r)/(0.25*r)

# Elements:
#  0: Uninfected
#  1: Initial infectious impulse
#  2: Number incubating
#  3: Number contagious 
#  4: Number observed

R=24.0

P=1e9

s=np.linspace(0.0,7.0*R,20) # Changing 20 to 40 makes too much difference... suspicious.
w=tri(s,7.0*R)
assert math.fabs(scipy.integrate.simps(w,s)-1.0) < 0.01


def sim(T):
    ts=[]
    ys=[]

    def f(t,y):

        # TODO: Not too convinced.  Number dropping out now should be the derivative of the number added a while ago.
        # But... the average derivative is just a gradient based on two end point values!  Bit more complicated if weighted though.
        d=np.linspace(t-14.0,t-7.0,29)
        tv=np.array(ts)
        yv=np.array(ys)
        yv2=yv[:,2]
        yv3=yv[:,3]
        w2=np.trapz(np.interp(d,tv,yv2),d)/7.0
        w3=np.trapz(np.interp(d,tv,yv3),d)/7.0

        print t,w2,w3
        
        i=(1.2*y[3]+y[1])*(y[0]/P)  
        return np.array([
            -i,
            -y[1],
            i-w2,
            w2-w3,
            0.05*i
        ])

    t=0.0
    y=np.array([P,1.0,0.0,0.0,0.0])

    while t<=T:
        ts.append(t)
        ys.append(y)

        s=scipy.integrate.solve_ivp(f,(t,t+1.0),y)
        y=s.y[:,-1]
        print y
        t=t+1.0

    return np.array(ts),np.array(ys)

ts,ys=sim(120)
        
plt.plot(ts,ys[:,0],color='black')
plt.plot(ts,ys[:,1],color='purple')
plt.plot(ts,ys[:,2],color='blue')
plt.plot(ts,ys[:,3],color='green')
plt.plot(ts,ys[:,4],color='red')
plt.yscale('symlog')
plt.grid(True)
plt.show()
