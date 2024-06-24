# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:57:33 2024

@author: Paul
"""
import sympy as sp
import numpy as np



def get_trace_eigenvals(matrix):
    eigenvals=matrix.eigenvals()
    squared_matrix= matrix*matrix
    trace=squared_matrix.trace().evalf()
    return trace,eigenvals

def calculate_avrg(Averiging_rate,sol_00,sol_11,sol_22,t_last_x_values):
    zero_vals = sol_00[-Averiging_rate:]
    one_vals=sol_11[-Averiging_rate:]
    two_vals=sol_22[-Averiging_rate:]
    # Compute the time interval T(x)
    T_x = t_last_x_values[-1] - t_last_x_values[0]
    integral_00 = np.trapz(zero_vals,t_last_x_values)/ T_x 
    #print(integral_00)
    integral_11 = np.trapz(one_vals,t_last_x_values)/ T_x 
    integral_22 = np.trapz(two_vals,t_last_x_values)/ T_x 
    
    averaged_vals=[integral_00 ,integral_11 ,integral_22]
    return averaged_vals

def calculate_variance(Averiging_rate,sol_00,sol_11,sol_22,averaged_vals,t_last_x_values):
    integral_00 ,integral_11 ,integral_22=averaged_vals
    zero_vals = sol_00[-Averiging_rate:]
    one_vals=sol_11[-Averiging_rate:]
    two_vals=sol_22[-Averiging_rate:]
    zero_vals = sol_00[-Averiging_rate:]
    one_vals=sol_11[-Averiging_rate:]
    two_vals=sol_22[-Averiging_rate:]
    
    zero_integrand=(zero_vals-integral_00)**2
    one_integrand=(one_vals-integral_11)**2
    two_integrand=(two_vals-integral_22)**2
    
    T_x = t_last_x_values[-1] - t_last_x_values[0]
    var_00 = np.trapz(zero_integrand,t_last_x_values)/T_x
    var_11 = np.trapz(one_integrand,t_last_x_values)/T_x
    var_22 = np.trapz(two_integrand,t_last_x_values)/T_x
    variances=[var_00,var_11,var_22]
    return variances



