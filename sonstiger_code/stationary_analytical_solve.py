# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:32:02 2024

@author: Paul
"""

# import sympy as sp
# import numpy as np
# # Definition der Symbole
# kappa,V, gamma, Gamma, g_0, Omega, eta, Delta_1,Delta_2 = sp.symbols('kappa V gamma Gamma g_0 Omega eta Delta_1 Delta_2')
# # Definition der Operatoren a und a^\dagger
# a, adagger = sp.symbols('a adagger')

# # Definition der Ketbras
# sigma00, sigma01, sigma02, sigma10, sigma11, sigma12, sigma20, sigma21, sigma22 = sp.symbols(
#     'sigma_00 sigma_01 sigma_02 sigma_10 sigma_11 sigma_12 sigma_20 sigma_21 sigma_22'
# )





# # Koeffizientenmatrix A
# A = np.array([
#     [-kappa / 2, 0,0,0,0,-1j*g_0*gamma,0,0,0,0,0], #a
#     [0, -kappa / 2,0,0,0,0,-1j*g_0*gamma,0,0,0,0], #a^\dagger
#     [1j*(gamma*g_0*sigma10), -1j*(gamma*g_0*sigma01),Gamma,0,0,-1j*g_0*gamma*adagger,1j*g_0*gamma*a,0,0,0,0], #sigma00
#     [1j*(gamma*g_0*sigma10), -1j*(gamma*g_0*sigma01),-Gamma,0,0,-1j*g_0*gamma*adagger,1j*g_0*gamma*a,0,0,-1j*Omega/2,1j*Omega/2], #sigma11
#     [0, 0,0,0,0,0,0,0,0,1j*Omega/2,-1j*Omega/2], #sigma22
#     [1j*gamma*g_0*(sigma11-sigma00), 0,1j*gamma*g_0*(-a),1j*gamma*g_0*a,0,-1j*(Delta_1+Gamma),0,1j*Omega/2,0,0,0], #sigma01
#     [0, -1j*gamma*g_0*(sigma11-sigma00),0,1j*gamma*g_0*(adagger),-1j*gamma*g_0*adagger,0,1j*(Delta_1+Gamma),0,-1j*Omega/2,0,0], #sigma10
#     [1j*gamma*g_0*sigma12, 0,0,0,-1j*2*V*sigma02,-1j*Omega/2,0,-1j*(Delta_2+2*V*sigma22),0,1j*gamma*g_0*a,0], #sigma02
#     [0, -1j*gamma*g_0*sigma21,0,0,1j*2*V*sigma20,0,1j*Omega/2,0,1j*(Delta_2+2*V*sigma22),0,-1j*gamma*g_0*adagger], #sigma20
#     [-1j*gamma*g_0*sigma20, 0,0,1j*Omega/2,-1j*(Omega/2-2*V*sigma12),0,0,-1j*gamma*g_0*a,0,0,-Gamma/2+1j*(Delta_2-Delta_1+2*V*sigma22)], #sigma12
#     [0, 1j*gamma*g_0*sigma02,0,-1j*Omega/2,1j*(Omega/2-2*V*sigma21),0,0,0,1j*gamma*g_0*adagger,-Gamma/2-1j*(Delta_2-Delta_1+2*V*sigma22),0], #sigma21
#     [0, 0,1,1,1,0,0,0,0,0,0],    
# ])

# # Ergebnismatrix B
# B = np.array([
#     eta,
#     eta,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     1
# ])









import sympy as sp
import numpy as np

# Defining the symbols
a, b, c, d, E, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t = sp.symbols('a b c d E f g h i j k l m n o p q r s t')

# Koeffizientenmatrix A
A = sp.Matrix([
    [a, b, c, d], 
    [f, g, h, i],
    [k, l, m, n],
    [p, q, r, s]
])

# Ergebnismatrix B
B = sp.Matrix([
    E,
    j,
    o,
    t
])

# Solving the equation A * X = B
#X = A.inv() * B

print(sp.det(A))
# Displaying the result
#sp.pprint(X)


# ahat=B0-N1/M1*A0
# fhat=B1-N1/M1*B1
# khat=F1-N1/M1*E1
# phat=J1-N1/M1*I1

# bhat=h*f-O1/M1*A0
# ghat=C1-O1/M1*B1
# lhat=G1-O1/M1*E1
# qhat=K1-O1/M1*I1
 
# chat=-h+f/M1*A0
# hhat=f/M1*B1
# mhat=-N+f/M1*E1
# rhat=f/M1*I1

# dhat=-R/M1*A0
# fhat=g-R/M1*B1
# nhat=f-R/M1*E1
# shat=R-R/M1*I1


# Ehat=C0+P1/M1*A0
# jhat=D1+P1/M1*A1
# ohat=L1+P1/M1*E1
# that=P1+P1/M1*I1



