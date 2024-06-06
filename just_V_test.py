import sympy as sp

# Definition der Variablen
a, a_dagger = sp.symbols('a a_dagger', complex=True)
ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = sp.symbols('ket00 ket01 ket10 ket11 ket22 ket21 ket12 ket20 ket02', complex=True)
kappa, gamma, eta, Gamma, delta_1, delta_2, Omega, V = sp.symbols('kappa gamma eta Gamma delta_1 delta_2 Omega V', real=True)

# Differentialgleichungen
da_dt = -kappa/2 * a - sp.I * (gamma * ket01) - eta
da_dagger_dt = sp.conjugate(da_dt)

dket00_dt = Gamma * ket11 + sp.I * gamma * (ket10 * a - ket01 * sp.conjugate(a))
dket01_dt = -Gamma/2 * ket01 + sp.I * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega/2 * ket02)
dket10_dt = sp.conjugate(dket01_dt)

dket11_dt = -Gamma * ket11 + sp.I * gamma * (ket01 * sp.conjugate(a) - ket10 * a) + sp.I * Omega/2 * (ket21 - ket12)
dket22_dt = sp.I * Omega / 2 * (ket12 - ket21)

dket21_dt = -Gamma/2 * ket21 + sp.I * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega/2 * (ket11 - ket22) + 2 * V * ket21 * ket22)
dket12_dt = sp.conjugate(dket21_dt)

dket02_dt = sp.I * (-delta_2 * ket02 - Omega/2 * ket01 - 2 * V * ket02 * ket22 + gamma * ket12 * a)
dket20_dt = sp.conjugate(dket02_dt)

# Stationäre Lösungen finden
steady_state = sp.solve([
    sp.Eq(da_dt, 0), sp.Eq(da_dagger_dt, 0), sp.Eq(dket00_dt, 0), sp.Eq(dket01_dt, 0), sp.Eq(dket10_dt, 0), 
    sp.Eq(dket11_dt, 0), sp.Eq(dket22_dt, 0), sp.Eq(dket21_dt, 0), sp.Eq(dket12_dt, 0), sp.Eq(dket02_dt, 0), sp.Eq(dket20_dt, 0)
], (a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02))

# Abweichungen einführen
delta_a, delta_adagger = sp.symbols('delta_a delta_adagger', complex=True)
delta_ket00, delta_ket01, delta_ket10, delta_ket11, delta_ket22, delta_ket21, delta_ket12, delta_ket20, delta_ket02 = sp.symbols('delta_ket00 delta_ket01 delta_ket10 delta_ket11 delta_ket22 delta_ket21 delta_ket12 delta_ket20 delta_ket02', complex=True)

a = steady_state[a] + delta_a
a_dagger = steady_state[a_dagger] + delta_adagger
ket00 = steady_state[ket00] + delta_ket00
ket01 = steady_state[ket01] + delta_ket01
ket10 = steady_state[ket10] + delta_ket10
ket11 = steady_state[ket11] + delta_ket11
ket22 = steady_state[ket22] + delta_ket22
ket21 = steady_state[ket21] + delta_ket21
ket12 = steady_state[ket12] + delta_ket12
ket20 = steady_state[ket20] + delta_ket20
ket02 = steady_state[ket02] + delta_ket02

# Linearisieren
linear_eqs = [
    sp.Eq(sp.diff(delta_a, 't'), -kappa/2 * delta_a - sp.I * (gamma * delta_ket01)),
    sp.Eq(sp.diff(delta_adagger, 't'), sp.conjugate(-kappa/2 * delta_a - sp.I * (gamma * delta_ket01))),
    sp.Eq(sp.diff(delta_ket00, 't'), Gamma * delta_ket11 + sp.I * gamma * (delta_ket10 * a - delta_ket01 * a_dagger)),
    sp.Eq(sp.diff(delta_ket01, 't'), -Gamma/2 * delta_ket01 + sp.I * (-delta_1 * delta_ket01 + gamma * (delta_ket11 * a - delta_ket00 * a) - Omega/2 * delta_ket02)),
    sp.Eq(sp.diff(delta_ket10, 't'), sp.conjugate(-Gamma/2 * delta_ket01 + sp.I * (-delta_1 * delta_ket01 + gamma * (delta_ket11 * a - delta_ket00 * a) - Omega/2 * delta_ket02))),
    sp.Eq(sp.diff(delta_ket11, 't'), -Gamma * delta_ket11 + sp.I * gamma * (delta_ket01 * a_dagger - delta_ket10 * a) + sp.I * Omega/2 * (delta_ket21 - delta_ket12)),
    sp.Eq(sp.diff(delta_ket22, 't'), sp.I * Omega / 2 * (delta_ket12 - delta_ket21)),
    sp.Eq(sp.diff(delta_ket21, 't'), -Gamma/2 * delta_ket21 + sp.I * (delta_2 * delta_ket21 - delta_1 * delta_ket21 - gamma * delta_ket20 * a + Omega/2 * (delta_ket11 - delta_ket22) + 2 * V * delta_ket21 * delta_ket22)),
    sp.Eq(sp.diff(delta_ket12, 't'), sp.conjugate(-Gamma/2 * delta_ket21 + sp.I * (delta_2 * delta_ket21 - delta_1 * delta_ket21 - gamma * delta_ket20 * a + Omega/2 * (delta_ket11 - delta_ket22) + 2 * V * delta_ket21 * delta_ket22))),
    sp.Eq(sp.diff(delta_ket02, 't'), sp.I * (-delta_2 * delta_ket02 - Omega/2 * delta_ket01 - 2 * V * delta_ket02 * delta_ket22 + gamma * delta_ket12 * a)),
    sp.Eq(sp.diff(delta_ket20, 't'), sp.conjugate(sp.I * (-delta_2 * delta_ket02 - Omega/2 * delta_ket01 - 2 * V * delta_ket02 * delta_ket22 + gamma * delta_ket12 * a)))
]

linear_eqs
