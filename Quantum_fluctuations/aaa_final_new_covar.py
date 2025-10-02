   # print("3. Differentialgleichungssystem wird gelöst...")
        # t_span = (0.0,20.0)
        # t_eval = np.linspace(*t_span, 1001)
        # Y0 = np.concatenate([np.real(m0), Sigma0.flatten()])
        # pprint(Matrix(Sigma0))
        # sol = solve_ivp(
        #     fun=lambda t, y: rhs_combined(t, y, params, g_func, w_func),
        #     t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45'
        # )
        # print("Lösung erfolgreich berechnet.")




        # diagnose_compatibility(sol,params,g_func)

        # # # ========================================================================
        # # # 5. POST-PROCESSING & KOMBINIERTER PLOT
        # # # ========================================================================
        # print("4. Berechne relevante Eigenwerte für G(t), W(t), M(t) und Sigma(t)...")

        # #--- Berechnung für G(t) und W(t) ---
        # min_eigenvalues_W = []
        # max_real_eigenvalues_G = [] # NEUE LISTE für G-Eigenwerte

        # for i in range(len(sol.t)):
        #     m_t = sol.y[:10, i]
        #     m_reordered = np.concatenate([m_t[2:], m_t[:2]])
            
        #     # Kleinster Eigenwert von W(t)
        #     W_t = w_func(*m_reordered)
        #     min_eigenvalues_W.append(np.min(np.linalg.eigvalsh(W_t)))

        #     # NEU: Größter Realteil der Eigenwerte von G(t)
        #     G_t = g_func(*m_reordered)
        #     eigenvalues_G = np.linalg.eigvals(G_t) # G ist nicht symmetrisch -> eigvals
        #     max_real_eigenvalues_G.append(np.max(np.real(eigenvalues_G)))

        # # # --- Berechnung für M(t) und Sigma(t) ---
        # #s_timeseries = build_symplectic_matrix_ts(sol)
        # s_timeseries = build_s_form_timeseries_2f(sol)
        # min_eigenvalue_trajectory_M = []
        # min_eigenvalues_Sigma = []

        # for i in range(len(sol.t)):
        #     C_t = sol.y[10:, i].reshape((10,10))
        #     sigma_t = C_t#0.5*(C_t + C_t.T)  # <- ab jetzt NUR Sigma_t benutzen
        #     min_eigenvalues_Sigma.append(np.min(np.linalg.eigvalsh(sigma_t)))
        #     s_t = s_timeseries[i]
        #     M_t = sigma_t + (1j / 2) * s_t
        #     min_eigenvalue_trajectory_M.append(np.min(np.linalg.eigvalsh(M_t)))
        #     #pprint(Matrix(sigma_t))    
        # pprint(Matrix(s_timeseries[-1]))
        # pprint(Matrix(Sigma0))
        # print(sol.y[:10, -1])
        # # --- Kombinierter Plot für alle vier Verläufe ---
        # print("5. Erstelle kombinierten Plot...")
        # plt.figure(figsize=(14, 8))
        
        # plt.plot(sol.t, max_real_eigenvalues_G, label='Größter Realteil Eig(G(t)) (Stabilität)', color='green', linestyle='-.') # NEUER PLOT
        # plt.plot(sol.t, min_eigenvalues_W, label='Kleinster Eig(W(t))', color='darkcyan', linestyle='--')
        # plt.plot(sol.t, min_eigenvalues_Sigma, label=r'Kleinster Eig($\Sigma(t)$)', color='orange')
        # plt.plot(sol.t, min_eigenvalue_trajectory_M, label=r'Kleinster Eig(M(t) = $\Sigma + \frac{i}{2}s$)', color='purple', linewidth=2)

        # plt.axhline(0, color='red', linestyle=':', linewidth=2, label='Referenzlinie (y=0)')
        # plt.title('Zeitentwicklung der relevanten Eigenwerte')
        # plt.xlabel('Zeit')
        # plt.ylabel('Wert des Eigenwerts / Realteils')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


        # # # ========================================================================
        # # # 6. WEITERE PLOTS (unverändert)
        # # # ========================================================================

        # # --- Plot 1: Alle Elemente der Kovarianzmatrix ---
        # print("Erstelle Plot: Zeitentwicklung der Kovarianzmatrix-Elemente...")
        # sigma_trajectories = sol.y[10:, :]
        # plt.figure(figsize=(12, 7))
        # plt.plot(sol.t, sigma_trajectories.T, alpha=0.5) 
        # plt.title('Zeitentwicklung aller 100 Elemente der Kovarianzmatrix Σ(t)')
        # plt.xlabel('Zeit')
        # plt.ylabel('Wert des Matrixelements')
        # plt.grid(True)
        # plt.show()
        # # print("Erstelle Plot: Zeitentwicklung der Kovarianzmatrix-Elemente...")
        # # singlets_traj = sol.y[:10, :]
        # # plt.figure(figsize=(12, 7))
        # # plt.plot(sol.t, singlets_traj.T, alpha=0.5) 
        # # plt.title('Zeitentwicklung aller 10')
        # # plt.xlabel('Zeit')
        # # plt.ylabel('Wert des Matrixelements')
        # # plt.grid(True)
        # # plt.show()


        # # --- Plot 2: Populationen rho_ii ---
        # print("Erstelle Plot: Zeitentwicklung der Populationen...")
        # def reconstruct_populations(sol_obj):
        #     x3_t = sol_obj.y[4, :]
        #     x8_t = sol_obj.y[9, :]
        #     sum00_11 = (2 + np.sqrt(3) * x8_t) / 3
        #     rho00_t = np.real((sum00_11 + x3_t) / 2)
        #     rho11_t = np.real((sum00_11 - x3_t) / 2)
        #     rho22_t = 1 - rho00_t - rho11_t
        #     return rho00_t, rho11_t, rho22_t

        # rho00, rho11, rho22 = reconstruct_populations(sol)
        # trace = rho00 + rho11 + rho22

        # plt.figure(figsize=(12, 7))
        # plt.plot(sol.t, rho00, label=r'$\rho_{00}(t)$')
        # plt.plot(sol.t, rho11, label=r'$\rho_{11}(t)$')
        # plt.plot(sol.t, rho22, label=r'$\rho_{22}(t)$')
        # plt.plot(sol.t, trace, '--', label='Summe (Spur)', linewidth=2)
        
        # plt.title('Zeitentwicklung der Populationen')
        # plt.xlabel('Zeit')
        # plt.ylabel('Population')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # # # print("\n\n" + "="*70)
        # # # print("DETAILLIERTE MATRIX-ANALYSE FÜR EINEN ZEITPUNKT")
        # # # print("="*70)
        # # # # Überprüfe die Matrizen am Ende der Simulation
        # # # get_and_check_matrices_at_time(t_target=sol.t[-1], sol=sol)

