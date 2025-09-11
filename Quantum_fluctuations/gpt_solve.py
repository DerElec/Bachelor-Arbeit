import gpt_try_on_covar as covar
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Numerische Parameter definieren
numeric_params = {
    sp.symbols("g0"): 1.0,
    sp.symbols("Delta1"): 1.0,
    sp.symbols("Delta2"): 1.0,
    sp.symbols("V"): -4.0,
    sp.symbols("gamma"): 1.0,
    sp.symbols("Omega"): 8.0,
    sp.symbols("kappa"): 1.0,
    sp.symbols("eta"): 1.0,
}

# 2. Das symbolische DGL-System einmalig erstellen
# state_vars ist die geordnete Liste der Symbole, z.B. [m1, m2, ..., m1m1, m1m2, ...]
# d_state_dt_sym ist die Liste der zugehörigen Ableitungen
state_vars, d_state_dt_sym = covar.build_ode_system(num_vars=10)

# 3. Die numerische Funktion für den Solver generieren
# Dies setzt die Parameter ein und kompiliert die DGLs zu einer schnellen Funktion.
solver_function = covar.create_numerical_ode_function(state_vars, d_state_dt_sym, numeric_params)

# 4. Anfangsbedingungen definieren
# Der Vektor y0 muss dieselbe Reihenfolge wie state_vars haben!
# y0 = [<m1>, <m2>, ..., <m10>, <m1m1>, <m1m2>, ..., <m10m10>]
# Hier als Beispiel: System startet im Vakuum (<m_i>=0, <m_i m_j>=0)
num_total_vars = 10 + 10*10 
y0 = np.zeros(num_total_vars)
# Vielleicht ist <m9m9> (q*q) oder ein anderer Wert am Anfang nicht null
# y0[10+88] = 1 # Beispiel: y0 für <m9m9> auf 1 setzen

# 5. Den Solver aufrufen
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 201)

solution = solve_ivp(
    fun=solver_function,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45'  # Eine robuste Methode
)

# 6. Ergebnisse plotten
# Index für <m1> ist 0, für <m2> ist 1, usw.
plt.figure(figsize=(12, 7))
plt.plot(solution.t, solution.y[0], label=r'$\langle m_1 \rangle$')
plt.plot(solution.t, solution.y[1], label=r'$\langle m_2 \rangle$')
plt.plot(solution.t, solution.y[8], label=r'$\langle m_9 \rangle$ (q)')

# Index für <m1m1> ist 10, für <m1m2> ist 11, usw.
# Index von <m_i m_j> ist 10 + (i-1)*10 + (j-1)
plt.plot(solution.t, solution.y[10 + 8*10 + 8], label=r'$\langle m_9 m_9 \rangle$ (Var q)', linestyle='--')

plt.title("Zeitentwicklung des Systems")
plt.xlabel("Zeit")
plt.ylabel("Erwartungswert / Korrelation")
plt.legend()
plt.grid(True)
plt.show()