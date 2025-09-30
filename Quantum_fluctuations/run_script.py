import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import time
import symplectic_matrix as symplect
import sympy as sp
# -------------------------------------------------------------------
# 1) RHS in ket-representation: returns dy/dt for 11-component y
# -------------------------------------------------------------------
def rhs_gellmann_qp_from_ket(t, y, params):
    """Compute derivatives da_dt, dρ_ij_dt for y = [a, ad, ρ00, ρ01, ρ10, ρ11, ρ22, ρ21, ρ12, ρ20, ρ02]."""
    # unpack state
    a, a_dagger = y[0], y[1]
    ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]
    # unpack parameters
    κ, γ, Γ, Ω, δ1, δ2, η, V = (params[k] for k in ('kappa','gamma','Gamma','Omega','delta1','delta2','eta','V'))

    # ladder dynamics
    da_dt        = -κ/2 * a - 1j*(γ*ket01) + η
    da_dagger_dt = np.conj(da_dt)

    # density-matrix amplitudes
    d00 = Γ*ket11 + 1j*γ*(ket10*a - ket01*a_dagger)
    d01 = -Γ/2*ket01 + 1j*(-δ1*ket01 + γ*(ket11*a - ket00*a) - Ω/2*ket02)
    d10 = np.conj(d01)

    d11 = -Γ*ket11 + 1j*γ*(ket01*a_dagger - ket10*a) + 1j*(Ω/2)*(ket21 - ket12)
    d22 = 1j*(Ω/2)*(ket12 - ket21)

    d21 = -Γ/2*ket21 + 1j*(δ2*ket21 - δ1*ket21 - γ*ket20*a + (Ω/2)*(ket11 - ket22) + 2*V*ket21*ket22)
    d12 = np.conj(d21)

    d02 = 1j*(-δ2*ket02 - Ω/2*ket01 - 2*V*ket02*ket22 + γ*ket12*a)
    d20 = np.conj(d02)

    return np.array([da_dt, da_dagger_dt, d00, d01, d10, d11, d22, d21, d12, d20, d02], dtype=complex)

# -------------------------------------------------------------------
# 2) convert_state: between 11-vector (ket) and 10-vector (Q,P,x1..x8)
# -------------------------------------------------------------------
def convert_state(y):
    """
    If len(y)==11, project to 10-vector [Q,P,x1..x8].
    If len(y)==10, reconstruct to 11-vector [a,ad,ρ00,ρ01,ρ10,ρ11,ρ22,ρ21,ρ12,ρ20,ρ02].
    """
    if len(y) == 11:
        # full → projected
        a, ad = y[0], y[1]
        rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho02, rho20 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[10], y[9]
        # Gell-Mann x's
        x1 = rho01 + rho10
        x2 = -1j*(rho01 - rho10)
        x3 = rho00 - rho11
        x4 = rho02 + rho20
        x5 = -1j*(rho02 - rho20)
        x6 = rho12 + rho21
        x7 = -1j*(rho12 - rho21)
        x8 = (rho00 + rho11 - 2*rho22)/np.sqrt(3)
        # quadratures
        Q = (a + ad)/np.sqrt(2)
        P = (a - ad)/(1j*np.sqrt(2))
        return np.array([Q, P, x1, x2, x3, x4, x5, x6, x7, x8], dtype=complex)

    elif len(y) == 10:
        # projected → full
        Q, P = y[0], y[1]
        x1, x2, x3, x4, x5, x6, x7, x8 = y[2:]
        # ladder
        a  = (Q + 1j*P)/np.sqrt(2)
        ad = np.conj(a)
        # density diag
        rho00 = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
        rho11 = 1/3 + 0.5*(-x3 + x8/np.sqrt(3))
        rho22 = 1 - rho00 - rho11
        # off-diags
        rho01 = (x1 + 1j*x2)/2
        rho02 = (x4 + 1j*x5)/2
        rho12 = (x6 + 1j*x7)/2
        rho10, rho20, rho21 = np.conj(rho01), np.conj(rho02), np.conj(rho12)
        return np.array([a, ad, rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02], dtype=complex)

    else:
        raise ValueError(f"Länge von y muss 10 oder 11 sein, got {len(y)}")

# -------------------------------------------------------------------
# 3) RHS in (Q,P,x)-representation using previous functions
# -------------------------------------------------------------------
def rhs_gellmann_qp_from_x(t, x, params):
    """Compute d[Q,P,x1..x8]/dt by converting to ket, calling rhs, then projecting."""
    # reconstruct full ket-vector
    y_full = convert_state(x)
    # compute dy/dt in ket
    dy = rhs_gellmann_qp_from_ket(t, y_full, params)
    # unpack
    da_dt, da_dagger_dt = dy[0], dy[1]
    d00, d01, d10, d11 = dy[2], dy[3], dy[4], dy[5]
    d22, d21, d12, d20, d02 = dy[6], dy[7], dy[8], dy[9], dy[10]
    # compute dx's
    dx1 =  d01 + d10
    dx2 = -1j*d01 + 1j*d10
    dx3 =  d00 - d11
    dx4 =  d02 + d20
    dx5 = -1j*d02 + 1j*d20
    dx6 =  d12 + d21
    dx7 = -1j*d12 + 1j*d21
    dx8 = (d00 + d11 - 2*d22)/np.sqrt(3)
    # quadrature derivatives
    dQ = (da_dt + da_dagger_dt)/np.sqrt(2)
    dP = (da_dt - da_dagger_dt)/(1j*np.sqrt(2))
    return np.array([dQ, dP, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8], dtype=complex)



# Covariance matrix DGl
def rhs_gellmann_qp_from_x(t, x, params):
    """Compute d[Q,P,x1..x8]/dt by converting to ket, calling rhs, then projecting."""
    # reconstruct full ket-vector
    y_full = convert_state(x)
    # compute dy/dt in ket
    dy = rhs_gellmann_qp_from_ket(t, y_full, params)
    # unpack
    da_dt, da_dagger_dt = dy[0], dy[1]
    d00, d01, d10, d11 = dy[2], dy[3], dy[4], dy[5]
    d22, d21, d12, d20, d02 = dy[6], dy[7], dy[8], dy[9], dy[10]
    # compute dx's
    dx1 =  d01 + d10
    dx2 = -1j*d01 + 1j*d10
    dx3 =  d00 - d11
    dx4 =  d02 + d20
    dx5 = -1j*d02 + 1j*d20
    dx6 =  d12 + d21
    dx7 = -1j*d12 + 1j*d21
    dx8 = (d00 + d11 - 2*d22)/np.sqrt(3)
    # quadrature derivatives
    dQ = (da_dt + da_dagger_dt)/np.sqrt(2)
    dP = (da_dt - da_dagger_dt)/(1j*np.sqrt(2))
    return np.array([dQ, dP, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8], dtype=complex)




# -------------------------------------------------------------------
# Example: solve in x-space and plot populations
# -------------------------------------------------------------------
if __name__ == '__main__':
    # define parameters
    params = dict(
        kappa=1.0, gamma=1.0, Gamma=2.0,
        Omega=8.0, delta1=1.0, delta2=1.0,
        eta=1.0, V=-1/2*((8/4)**2+1)
    )

    # time span
    t0, t_end = 0.0, 2000.0

    # initial ket-state (ρ00=1)
    y0 = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    # project to x-space
    x0 = convert_state(y0)

    # solve ODE in x-space
    start = time.time()
    sol = solve_ivp(lambda t, x: rhs_gellmann_qp_from_x(t, x, params),
                    (t0, t_end), x0, method='RK45',
                    atol=1e-8, rtol=1e-8)
    dauer = time.time() - start

    # reconstruct populations over time
    t_vals = sol.t
    rho00 = np.empty_like(t_vals, dtype=float)
    rho11 = np.empty_like(t_vals, dtype=float)
    rho22 = np.empty_like(t_vals, dtype=float)
    for i, xi in enumerate(sol.y.T):
        # unpack xi = [Q,P,x1..x8]
        Q, P, x1, x2, x3, x4, x5, x6, x7, x8 = xi
        sum00_11 = (2 + np.sqrt(3)*x8)/3
        rho00[i] = np.real((sum00_11 + x3)/2)
        rho11[i] = np.real((sum00_11 - x3)/2)
        rho22[i] = 1 - rho00[i] - rho11[i]

    # plot populations
    plt.figure()
    plt.plot(t_vals, rho00, label=r'$\rho_{00}$')
    plt.plot(t_vals, rho11, label=r'$\rho_{11}$')
    plt.plot(t_vals, rho22, label=r'$\rho_{22}$')
    plt.plot(t_vals, rho00+rho11+rho22, '--', label='Spur')
    plt.xlabel('Zeit')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.title('Populationsdynamik')
    plt.tight_layout()
    plt.show()

    # print(f"Ausführungsdauer: {dauer:.2f} Sekunden")
    # x = sol.y[:, -1]   # Länge 10: [Q, P, x1, x2, x3, x4, x5, x6, x7, x8]
    # m = x[2:]
    # C_num_sympy = symplect.get_S_matrix_gellman(m)
    # S = np.array(C_num_sympy.tolist(), dtype=complex)
    # final_result,J,R=symplect.transform_complex_S(S)
    # JRSRJ=symplect.expand_to_10x10(1j*final_result)
    # R=symplect.expand_to_10x10(R)
    # J=symplect.expand_to_10x10(J)
    # sp.pprint(sp.Matrix(symplect.expand_to_10x10_sym(final_result)))
    # sp.pprint(sp.Matrix(J))
    # sp.pprint(sp.Matrix(R))























    