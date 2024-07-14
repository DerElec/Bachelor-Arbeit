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

def calculate_avrg_vars(Averiging_rate,arr_of_sols,t_last_x_values):
    '''For a given Averiging rate takes an array of solutions and the last x values of the time to calculate the variances and averages of given values in the array of solution'''
    averaged_vals=[]
    variances=[]
    for sol in arr_of_sols:
        vals = sol[-Averiging_rate:]
        T_x = t_last_x_values[-1] - t_last_x_values[0]
        integral_sol = np.trapz(vals,t_last_x_values)/ T_x 
        averaged_vals.append(integral_sol)
        integrand_variances = (vals-integral_sol)**2
        T_x = t_last_x_values[-1] - t_last_x_values[0]
        variance= np.trapz(integrand_variances,t_last_x_values)/T_x
        variances.append(variance)
        
    return averaged_vals,variances

