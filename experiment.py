#!/usr/bin/env python

from sympy.abc import u,v,w,t
from sympy import Function, dsolve, Eq, Derivative, symbols

y=Function('y')
z=Function('z')

eq=(Eq(Derivative(z(t),t),v*y(t)-u*z(t)),Eq(Derivative(y(t),t),-v*y(t)+w*z(t)))
dsolve(eq)
