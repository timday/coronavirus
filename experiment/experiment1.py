#!/usr/bin/env python

from sympy import Function, dsolve, Eq, Derivative, symbols
import numpy as np
import matplotlib.pyplot as plt

y=Function('y')  # Contagious
u,v,w=symbols('u,v,w',real=True)  # u:contagion rate, v: infection-contagion delay, w: recovery rate 
t=symbols('t',real=True)
T=symbols('T',real=True)    # Time at which y(T)=1

eq=Eq(Derivative(y(t),t),u*y(t-v)-w*y(t))
s=dsolve(eq,ics={y(T):1})  

print s  # Fail: has some integral in it, unsurprisingly.
print

fn=s.rhs.subs(u,0.2).subs(w,0.1).subs(T,-30).subs(v,7.0)
print fn

tr=np.arange(-30,60)
plt.plot(tr,[fn.evalf(subs={t:tv}) for tv in tr],label='Contagious',color='red')
plt.yscale('symlog')
plt.legend()
plt.show()
