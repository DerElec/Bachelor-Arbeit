# -*- coding: utf-8 -*-
"""
Created on Sun May 12 23:30:38 2024

@author: holli
"""
kappa, gamma, Gamma, Omega, delta_2, delta_1, V=1,1,1,1,1,1,1

from sympy import symbols, Function, hessian
import numpy as np
from scipy.linalg import solve

def f(x):
    a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21=x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10]

    return np.array([
        -kappa/2*a_dagger+1j*gamma*psi10,
        Gamma*psi11-1j*gamma*(psi01*a_dagger-psi10*a),
        -Gamma*psi11-1j*gamma*(psi10*a-psi01*a_dagger)-1j*Omega/2*(psi21-psi12),
        -1j*Omega/2*(psi21-psi12),
        -Gamma/2*psi12-1j*(delta_2-delta_1)*psi12+1j*gamma*psi02*a_dagger-1j*Omega/2*(psi11-psi22)+2*V*psi12*psi22,
        -Gamma/2*psi10-1j*(-delta_1*psi10+gamma*(psi11*a_dagger-psi00*a)+Omega/2*(-psi20)),
        -1j*(delta_2*psi02+Omega/2*psi01+2*V*psi02*psi22-gamma*psi12*a),
        
        -kappa/2 * a - 1j * gamma * psi01,
        Gamma * psi11 +  1j * gamma * (psi10 * a - psi01 * a_dagger),
        -Gamma * psi11 + 1j * gamma * (psi01 * a_dagger - psi10 * a) + 1j * Omega/2 * (psi12 - psi21),
        1j * Omega/2 * (psi12 - psi21),
        -Gamma/2 * psi21 + 1j * (delta_2 - delta_1) * psi21 - 1j * gamma * psi20 * a + 1j * Omega/2 * (psi11 - psi22) + 2 * V * psi21 * psi22,
        -Gamma/2 * psi01 + 1j * (-delta_1 * psi01 + gamma * (psi11 * a - psi00 * a_dagger) + Omega/2 * (-psi02)),
        1j * (delta_2 * psi20 + Omega/2 * psi10 + 2 * V * psi20 * psi22 - gamma * psi21 * a_dagger)
    ])

def jacobian(F,a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21):
    return hessian(F,(a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21))







def newton_raphson(x0, tol=1e-6, max_iter=100):
    a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21 = symbols('a a_dagger psi00 psi11 psi22 psi01 psi10 psi02 psi20 psi12 psi21')
    x = x0
    for _ in range(max_iter):
        J = jacobian(f(x),a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21)
        print(J)
        F = f(x)
        delta = solve(J, -F)
        x = x + delta
        if np.linalg.norm(delta) < tol:
            return x
    return x

# Anfangsschätzung
x0 = np.array([1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
solution = newton_raphson(x0)

print("Lösung des Systems:", solution)