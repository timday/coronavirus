#!/usr/bin/env python

from sympy import Function, dsolve, Eq, Derivative, symbols

z=Function('z')
y=Function('y')
u,v,w,t=symbols('u,v,w,t',real=True)

eq=( Eq(Derivative(z(t),t),v*y(t)-u*z(t)) , Eq(Derivative(y(t),t),w*z(t)-v*y(t)) )
dsolve(eq)
