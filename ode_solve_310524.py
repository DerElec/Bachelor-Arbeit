import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import tracing as tr
import sympy as sp
import os
# def time_trac(function):
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()  # Start der Zeitmessung
#         result = function(*args, **kwargs)  # Funktion ausführen
#         end_time = time.perf_counter()  # Ende der Zeitmessung
#         elapsed_time = end_time - start_time  # Berechnung der verstrichenen Zeit
#         print(f"Die Ausführungsdauer von '{function.__name__}' betrug: {elapsed_time:.4f} Sekunden ")
#         return result
#     return wrapper
start_time = time.time()
def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start der Zeitmessung
        result = func(*args, **kwargs)  # Funktion ausführen
        end_time = time.perf_counter()  # Ende der Zeitmessung
        elapsed_time = end_time - start_time  # Berechnung der verstrichenen Zeit
        
        # Speichern der Zeit in einer .txt Datei
        with open('execution_times.txt', 'a') as file:
            file.write(f"Die Ausführungsdauer von '{func.__name__}' betrug: {elapsed_time:.4f} Sekunden\n")
        
        return result
    return wrapper

results = []
results_full = []
#
#for lt in np.arange(-1,0,1):
for delta_2 in np.arange(8,10,0.5):    
    for Omega in np.arange(5,10,0.5):  
        #print(delta_2)
        # Parameter for dgl    
        kappa = 1 #cavity loss rate0
        gamma = 1 #rate from cavity and atom coupling
        Gamma = 2 #Decay rate from first excited state to ground
        #Omega = 2 #Laser atom coupling constant
        #Omega = 2 #Laser atom coupling constant
        delta_1 = 1 #Detuning between first excited state and cavity-Pump detuning
        #delta_2 = 3 #Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field )
        eta=1
        
        #V = 0.2 #Atom-Atom coupling constant
        V=-delta_2*((Omega*kappa/(4*eta*gamma))**2+1)/2
        vals=[kappa,gamma,Gamma,Omega,delta_1,delta_2,eta,V]
        #V=-3.25
        #V=-delta_2/2*((Omega*kappa)**2/(16*(eta*gamma)**2)+1)
        print(f'kappa={kappa},gamma={gamma},Gamma={Gamma},V={V},Omega={Omega},eta={eta},delta_1,2={delta_1},{delta_2}')
        #print(f'V = {V}')
        #print(f'Omega = {Omega}')
        T=2000
         #Time 
        
        ########################################
        #stationary state 
        def get_constants(kappa,gamma,Omega,eta):
            #c_2 = 1 / np.sqrt((Omega**2 * kappa**2) / (16 * eta**2 * gamma**2 ) + 1)
            
            c_1 = (-Omega * kappa) / (4 * eta *  gamma) * np.sqrt(1 / ((Omega**2 * kappa**2) / (16 * eta**2 * gamma**2 ) + 1))
            c_2=np.sqrt(1-c_1*np.conj(c_1))
            return c_1,c_2
            
        c_1,c_2= get_constants(kappa,gamma,Omega,eta)
        #Random testing
        a0= 0
        a_dagger_0=0
        psi00 = 0
        psi11 = 0.0 + 0j
        psi22 = 1
        psi20 = 0
        psi02 = 0
        psi10 = 0.0 + 0j
        psi01 = 0.0 + 0j
        psi21=0.0+0j
        psi12=0.0+0j
        startcond=[a0,a_dagger_0,psi00,psi11,psi22,psi10,psi01,psi21,psi12,psi20,psi02]
        ##################################################
        #Random testing
        # a0= 2*eta/kappa+0j
        # a_dagger_0=2*eta/kappa+0j
        # psi00 = 0
        # psi11 = 0.0 + 0j
        # psi22 = 1
        # psi20 = -Omega*kappa*delta_2/(8*eta*gamma*V)
        # psi02 = 4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1)
        # psi10 = 0.0 + 0j
        # psi01 = 0.0 + 0j
        # psi21=0.0+0j
        # psi12=0.0+0j
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
        # psi02 = np.conj(psi20)
        # #other*(4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1))
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
        #print(psi20,psi02)
        # Define the elements of the density matrix rho
    
        
        
        #print(psi00+psi11+psi22)
        #@time_trac
        #@time_wrapper
        def dydt(t, y):
            # Zerlegung der Zustandsvariablen
            a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y
        
            #Differentialgleichungen für die Mittelwerte der Operatoren
            da_dt = -kappa/2 *a - 1j*( gamma * ket01)+eta
            #da_dagger_dt = -kappa/2 * a_dagger + 1j * gamma * ket10
            da_dagger_dt =np.conj(da_dt)
            
            dket00_dt = +Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
            
            dket01_dt = -Gamma/2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega/2 * ket02)
            #dket10_dt = -Gamma/2 * ket10 - 1j * (- (omega_1 - omega_c) * ket10 + gamma * (ket11_k * a_dagger - ket00 * a) - Omega/2 * ket20)
            dket10_dt =np.conj(dket01_dt)
            
            dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega/2 * (ket21 - ket12)
            dket22_dt = 1j * Omega / 2 * (ket12 - ket21)
            
            dket21_dt = -Gamma/2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega/2 * (ket11 - ket22) + 2*V * ket21*ket22)
            dket12_dt = np.conj(dket21_dt)
            
            dket02_dt= 1j*(-delta_2*ket02-Omega/2*ket01-2*V*ket02*ket22+gamma *ket12 *a)
            dket20_dt=np.conj(dket02_dt)
            
    
            return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt,dket20_dt,dket02_dt]
        
        
        t_eval = np.linspace(0, T, 10000)  
        y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12,psi20,psi02]  # Anfangsbedingungen, vereinfachte Anfangszustände für Nicht-Diagonalelemente
        
        #@time_wrapper
        def solver(y0):
            # sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval#, method='RK45', rtol=1e-12, atol=1e-15)
            #                 , method='RK45', rtol=1e-12, atol=1e-15)
            sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval#, method='RK45', rtol=1e-12, atol=1e-15)
                            , method='DOP853', rtol=1e-13, atol=1e-16)
            return sol
        sol=solver(y0)
    
        
        
        # plt.plot(sol.t, abs(sol.y[0]), label='a')
        # plt.plot(sol.t, abs(sol.y[1]), label='a^dagger')
        
        # #übergänge
        # plt.plot(sol.t, abs(sol.y[3]), label='<0|1>')
        # plt.plot(sol.t, abs(sol.y[4]), label='<1|0>')
        # plt.plot(sol.t, abs(sol.y[7]), label='<2|1>')
        # plt.plot(sol.t, abs(sol.y[8]), label='<1|2>')
        # plt.plot(sol.t, abs(sol.y[9]), label='<2|0>')
        # plt.plot(sol.t, abs(sol.y[10]), label='<0|2>')
        
        #adjoint zustände 
        plt.plot(sol.t, sol.y[2], label='<0|0>')
        plt.plot(sol.t, sol.y[5], label='<1|1>')
        plt.plot(sol.t, sol.y[6], label='<2|2>')
        # print(sol.y[2][0],sol.y[5][0],sol.y[6][0])
        # print(sol.y[2][-1],sol.y[5][-1],sol.y[6][-1])
        
        ######################################
        plt.plot(sol.t, sol.y[2]+sol.y[5]+sol.y[6])
        
        
        plt.xlabel('Time in arbitrary units')
        plt.ylabel('Value of expectation value')
        plt.legend()
        plt.title(f'kappa={kappa},gamma={gamma},Gamma={Gamma},V={V},Omega={Omega},eta={eta},delta_1,2={delta_1},{delta_2}')

        plt.show()
        ########################
        # Save the initial and final values
        final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
        final_values_full = [sol.y[2], sol.y[5], sol.y[6]]
        results.append([V] + final_values)
        
        #print([sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]])
       
        
       
        rho_unstationary=sp.Matrix([
            [sol.y[6][-1] ,sol.y[7][-1] , sol.y[9][-1] ],
            [sol.y[8][-1], sol.y[5][-1],sol.y[4][-1] ],
            [sol.y[10][-1], sol.y[3][-1],sol.y[2][-1] ]
        ])
        trace,eigenvals=tr.get_trace_eigenvals(rho_unstationary)
        for i in eigenvals:
            if sp.re(i)<-np.e**(-12):
                break
                print(f"problematic, eigenvalue is {i}")
        #print(f"Purity is {trace}") 
        
        
        
        ##################################################################################
        #Averiging results
        Averiging_rate=50
        averaged_vals=tr.calculate_avrg(Averiging_rate,sol.y[2],sol.y[5],sol.y[6])
        
        ##################################################################################
        #Calculating variance
    
        variances=tr.calculate_variance(Averiging_rate, sol.y[2],sol.y[5],sol.y[6], averaged_vals)
        #print(variances)
        ##################################################################################
        
        results_full.append([V] +averaged_vals #final_values_full
                            +[eigenvals]+[trace]+[vals]+[startcond]+[variances])
        
    df_new = pd.DataFrame(results, columns=['V', '<0|0>', '<1|1>', '<2|2>'])
    df_full_new = pd.DataFrame(results_full, columns=['V', '<0|0>', '<1|1>', '<2|2>','eigenvals','purity','additional params','startcond','Variances'])
    # Save to Excel
    
    
    # Check if the file exists
    if os.path.exists('results_3.pkl'):
        df_existing = pd.read_pickle('results_3.pkl')
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    if os.path.exists('results_full_3.pkl'):
        df_full_existing = pd.read_pickle('results_full_3.pkl')
        df_full_combined = pd.concat([df_full_existing, df_full_new], ignore_index=True)
    else:
        df_full_combined = df_full_new
    
    # Save the combined DataFrames
    df_combined.to_pickle('results_3.pkl')
    df_full_combined.to_pickle('results_full_3.pkl')
    
    
    
    
    
    # df.to_excel('results_3.xlsx', index=False, engine='openpyxl')
    
    # df_full.to_pickle('results_full_3.pkl')
    
    
    
    
    
    
    
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time  # Laufzeit berechnen
    print(f"Die Laufzeit der main-Funktion beträgt: {elapsed_time} Sekunden")
    #print("DataFrames wurden erfolgreich gespeichert.")
