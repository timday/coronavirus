#!/usr/bin/env python

from sympy import Function, dsolve, Eq, Derivative, symbols
import numpy as np
import matplotlib.pyplot as plt

z=Function('z')  # Infected
y=Function('y')  # Contagious
x=Function('x')  # Observed
k,u,v,w=symbols('k,u,v,w',real=True)  # k:Observation rate, u:contagion rate, v: infection-contagion conversion rate, w: recovery rate 
t=symbols('t',real=True)
n0=symbols('n0',real=True)  # Observed at t(0)
T=symbols('T',real=True)    # Time at which y(T)=1

eq=[
    Eq(Derivative(z(t),t),u*y(t)-v*z(t)),
    Eq(Derivative(y(t),t),v*z(t)-w*y(t)),
    Eq(Derivative(x(t),t),k*(v*z(t)-w*y(t)))
]
s=dsolve(eq,ics={x(T):0,y(T):1,z(T):0})

print s
print
print s[2]
print

fns=map(lambda f: f.subs(u,0.8).subs(v,0.1).subs(w,0.1).subs(T,-60).rhs,s)
print s[0].lhs,fns[0]
print s[1].lhs,fns[1]

kv=(n0/fns[1]).subs(t,0).subs(n0,300)  # Scale factor to get right observed number at t=0
print kv
fns[2]=fns[2].subs(k,kv)

print s[2].lhs,fns[2]

tr=np.arange(-60,60)
plt.plot(tr,[fns[0].evalf(subs={t:tv}) for tv in tr],label='Infected',color='blue')
plt.plot(tr,[fns[1].evalf(subs={t:tv}) for tv in tr],label='Contagious',color='green')
plt.plot(tr,[fns[2].evalf(subs={t:tv}) for tv in tr],label='Observed',color='red')
plt.yscale('symlog')
plt.legend()
plt.show()
