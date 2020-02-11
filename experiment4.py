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

def f(t,y):
    i=(2.0*y[3]+y[1])*(y[0]/P)
    return np.array([
        -i,
        -y[1],
        i-y[2],
        y[2]-y[3],
        0.05*i
    ])

def sim(T):

    t=0.0
    y=np.array([P,1.0,0.0,0.0,0.0])

    ts=[]
    ys=[]
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
