# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:57:33 2024

@author: Paul
"""
import sympy as sp
kappa = 1#cavity loss rate0
gamma = 2 #rate from cavity and atom coupling
Gamma = 2 #Decay rate from first excited state to ground
#Omega = 3 #Laser atom coupling constant
#Omega = 1.5 #Laser atom coupling constant
delta_1 = 90 #Detuning between first excited state and cavity-Pump detuning
delta_2 = 2 #Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field )
eta=1
Omega=3

#V = 0.2 #Atom-Atom coupling constant
#V=-delta_2*((Omega*kappa/(4*eta*gamma))**2+1)/2

V=-50
#Random testing
a0= 1
a_dagger_0=1
psi00 = 0
psi11 = 0.0 + 0j
psi22 = 1
psi20 = 0
psi02 = 2
psi10 = 1.0 + 0j
psi01 = 0.0 + 0j
psi21=0.0+0j
psi12=-1+0j
####################################
#Stationary state# perturbed
# eps=0.05
# other=1-eps
# a0= -2*eta/kappa+0j
# a_dagger_0=-2*eta/kappa+0j
# psi00 = other*(delta_2/(2*V)+1)+eps/2
# psi11 = 0 +0j
# psi22 = -other*delta_2/(2*V)+eps/2
# psi20 = other*(-Omega*kappa*delta_2/(8*eta*gamma*V))
# psi02 = other*(4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1))
# psi10 = 0.0 + 0j
# psi01 = 0.0 + 0j
# psi21=0.0+0j
# psi12=0.0+0j
#####################################
#Stationary state# unperturbed
# a0= -2*eta/kappa+0j
# a_dagger_0=-2*eta/kappa+0j
# psi00 = (delta_2/(2*V)+1)
# psi11 = 0 +0j
# psi22 = -delta_2/(2*V)
# psi20 = (-Omega*kappa*delta_2/(8*eta*gamma*V))
# psi02 = (4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1))
# psi10 = 0.0 + 0j
# psi01 = 0.0 + 0j
# psi21=0.0+0j
# psi12=0.0+0j


psi00*psi00



# Define symbolic placeholders for the wavefunction components
psi00_sym = sp.symbols('psi00')
psi22_sym = sp.symbols('psi22')
psi20_sym = sp.symbols('psi20')
psi02_sym = sp.symbols('psi02')

# Assign values to the placeholders
psi00_val = psi00
psi22_val = psi22
psi20_val = psi20
psi02_val = psi02



def get_trace_eigenvals(matrix):
    eigenvals=matrix.eigenvals()
    squared_matrix= matrix*matrix
    trace=squared_matrix.trace().evalf()
    return trace,eigenvals



# # Calculate eigenvalues
# eigenvalues = rho_stationary.eigenvals()

# # Calculate rho^2
# rho_squared_stationary = rho_stationary * rho_stationary


# # Calculate the trace of rho^2
# trace_rho_squared_stationary = rho_squared_stationary.trace()

# Display the trace
#for i in rho_stationary.eigenvals()

