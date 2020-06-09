# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:26:55 2020

@author: luis andres arias 17145
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sympy import init_printing
from scipy.integrate import quad
from scipy.integrate import quad
init_printing()

"""

dx= np.linspace(0,1,21)
dy= np.linspace(0,1,21)
dx,dy

X,Y= np.meshgrid(dx,dy)

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
for i in range(len(dx)):
    for j in range(len(dy)):
        ax.scatter(dx[i],dy[j], color='b')
plt.show()

def integrand(y,a):
    return np.arctan(y/a)

a=1
b=1

V0,err= quad(integrand,0,b, args=a)

def f(x,y,k):
    return (2*V0)*( (np.sinh(k*np.pi*x)*np.sin(k*np.pi*y))/(np.sinh(k*np.pi)) )

Z= f(X,Y,1)

Z= f(X,Y,2)
plt.contourf(X, Y, Z, 40)
plt.show()

def PlotearZ(n):
    Zres= 0
    for i in range(n):
        Zres= Zres+f(X,Y,i+1)
    plt.contourf(X, Y, Zres, 50);
    plt.show()
    
    
PlotearZ(1)

dx= np.linspace(-1,1,41)
dy= np.linspace(0,1,21)
dx,dy
X,Y= np.meshgrid(dx,dy)

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
for i in range(len(dx)):
    for j in range(len(dy)):
        ax.scatter(dx[i],dy[j], color='b')
plt.show()

def f2(x,y,k):
    
    arr1= np.sin(k*np.pi*y)*np.cosh(k*np.pi*x)
    aba1= np.cosh(k*np.pi)
    
    parte1=arr1/aba1
    
    
    var=(-1)**k
    arrint= 12 * k * np.pi - 7* np.pi**3 *k**3
    abaint= np.pi**4 * k**4
    cnte= 5/(k*np.pi)
    
    num=var*arrint/abaint+cnte
    
    
    
    return 2*((np.sin(k*np.pi*y)*np.cosh(k*np.pi*x))/(np.cosh(k*np.pi)))*((-1)**k *((12 * k * np.pi - 7* np.pi**3 *k**3)/(np.pi**4 * k**4))+(5/(k*np.pi)))

Z= f2(X,Y,1)
plt.contourf(X, Y, Z, 50)
plt.show()

def PlotearZ2(n):
    Zres= 0
    for i in range(n):
        Zres= Zres+f2(X,Y,i+1)
    plt.contourf(X, Y, Zres, 50);
    plt.show()

PlotearZ2(20)



"""






maxIter = 500
lenX = lenY = 20 
delta = 1
Tguess = 5

colorinterpolation = 50
colourMap = plt.cm.jet

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

#condiciones de frontera
Ttop = 0
Tbottom = 0
Tleft = 0
Tright = np.arctan(Y[:,:1]/a)

# Set array size and set the interior value with Tguess
T = np.empty((lenX, lenY))
T.fill(Tguess)


T[(lenY-1):, :] = Ttop
#T[:1, :] = Tbottom
T[:, (lenX-1):] = Tright
T[:, :1] = Tleft



print("Por favor espere")
for iteration in range(0, maxIter):
    for i in range(0, lenX-1, delta):
        for j in range(1, lenY-1, delta):
            T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])

print("Iteraciones terminadas")


plt.title("Contour of Temperature")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)


plt.colorbar()


plt.show()

print("")









"""







X, Y = np.meshgrid(np.arange(-lenX, lenX), np.arange(0, lenY))

#condiciones de frontera
Ttop = 0
Tbottom = 0
Tleft = Tright = 2*(Y[:,:1])**3+5


T = np.empty((lenX, 2*lenY))
T.fill(Tguess)

T[(lenY-1):, :] = Ttop
#T[:1, :] = Tbottom
T[:, (2*lenX-1):] = Tright
T[:, :1] = Tleft

print("Por favor espere")
for iteration in range(0, maxIter):
    for i in range(0, lenX-1, delta):
        for j in range(1, 2*lenY-1, delta):
            T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])

print("Iteraciones terminadas")


plt.title("Contour of Temperature")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)


plt.colorbar()

plt.show()

print("")
"""