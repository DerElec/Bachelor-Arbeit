
def create_all():
    import sympy as sp
    import symplectic_matrix as symplect
    import numpy as np
    # 1) Declare symbolic parameters and constants
    kappa, gamma, Gamma, Omega, delta1, delta2, eta, V = sp.symbols(
        "kappa gamma Gamma Omega delta1 delta2 eta V", real=True
    )
    sqrt2 = sp.sqrt(2)
    sqrt3 = sp.sqrt(3)

    # 2) Time and functions
    t = sp.symbols("t", real=True)
    q = sp.Function("q")(t)
    p = sp.Function("p")(t)
    x1 = sp.Function("x1")(t)
    x2 = sp.Function("x2")(t)
    x3 = sp.Function("x3")(t)
    x4 = sp.Function("x4")(t)
    x5 = sp.Function("x5")(t)
    x6 = sp.Function("x6")(t)
    x7 = sp.Function("x7")(t)
    x8 = sp.Function("x8")(t)

    # 3) Ladder operators
    a = (q + sp.I * p) * sqrt2
    a_dag = (q - sp.I * p) * sqrt2

    # 4) Density matrix elements
    rho00 = sp.Rational(1, 3) + sp.Rational(1, 2) * (x3 + x8 / sqrt3)
    rho11 = sp.Rational(1, 3) + sp.Rational(1, 2) * (-x3 + x8 / sqrt3)
    rho22 = sp.Rational(1, 3) - x8 / sqrt3
    rho01 = (x1 + sp.I * x2) / 2
    rho10 = sp.conjugate(rho01)
    rho02 = (x4 + sp.I * x5) / 2
    rho20 = sp.conjugate(rho02)
    rho12 = (x6 + sp.I * x7) / 2
    rho21 = sp.conjugate(rho12)

    # 5) ODEs
    da_dt = -(kappa / 2) * a - sp.I * gamma * rho01 + eta
    da_dag_dt = sp.conjugate(da_dt)
    d_rho00_dt = Gamma * rho11 + sp.I * gamma * (rho10 * a - rho01 * a_dag)
    d_rho01_dt = -(Gamma / 2) * rho01 + sp.I * (
        -delta1 * rho01 + gamma * (rho11 * a - rho00 * a) - Omega / 2 * rho02
    )
    d_rho10_dt = sp.conjugate(d_rho01_dt)
    d_rho11_dt = -Gamma * rho11 + sp.I * gamma * (rho01 * a_dag - rho10 * a) + sp.I * (
        Omega / 2
    ) * (rho21 - rho12)
    d_rho22_dt = sp.I * (Omega / 2) * (rho12 - rho21)
    d_rho21_dt = -(Gamma / 2) * rho21 + sp.I * (
        (delta2 - delta1) * rho21 - gamma * rho20 * a + (Omega / 2) * (rho11 - rho22) + 2 * V * rho21 * rho22
    )
    d_rho12_dt = sp.conjugate(d_rho21_dt)
    d_rho02_dt = sp.I * (
        -delta2 * rho02 - Omega / 2 * rho01 - 2 * V * rho02 * rho22 + gamma * rho12 * a
    )
    d_rho20_dt = sp.conjugate(d_rho02_dt)

    # 6) Gell-Mann derivatives
    dx1 = d_rho01_dt + d_rho10_dt
    dx2 = -sp.I * d_rho01_dt + sp.I * d_rho10_dt
    dx3 = d_rho00_dt - d_rho11_dt
    dx4 = d_rho02_dt + d_rho20_dt
    dx5 = -sp.I * d_rho02_dt + sp.I * d_rho20_dt
    dx6 = d_rho12_dt + d_rho21_dt
    dx7 = -sp.I * d_rho12_dt + sp.I * d_rho21_dt
    dx8 = (d_rho00_dt + d_rho11_dt - 2 * d_rho22_dt) / sqrt3

    # 7) Quadrature derivatives
    dq_dt = (da_dt + da_dag_dt) / sqrt2
    dp_dt = (da_dt - da_dag_dt) / (sp.I * sqrt2)

    # 8) Assemble f(x,t)
    dxdt = [dq_dt, dp_dt, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8]

    # Output LaTeX code component-wise
    names = ["q", "p", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8"]
    #for name, expr in zip(names, dxdt):
    #    print(f"\\frac{{d{name}}}{{dt}} &= {sp.latex(expr)} \\\\")

    import sympy as sp

    def single_column_matrix(eq_list, vars_list, rule='min'):
        """
        Build a matrix A and vector b such that
            eq_list ≡ A * vars_list + b
        but assign every nonlinear term to EXACTLY ONE column.
        rule = 'min'  → take the variable with the lowest index
        rule = 'max'  → ... highest index
        """
        n = len(eq_list); m = len(vars_list)
        A = sp.zeros(n, m)
        b = sp.zeros(n, 1)

        for i, expr in enumerate(eq_list):
            expr = sp.expand(expr)
            for term in expr.as_ordered_terms():

                vars_in_term = [v for v in vars_list if term.has(v)]

                if not vars_in_term:                 # purely constant
                    b[i] += term
                    continue

                # choose one variable according to the rule
                v = min(vars_in_term, key=vars_list.index) if rule=='min' \
                    else max(vars_in_term, key=vars_list.index)

                coeff = sp.expand(term / v)          # remove that variable once
                col   = vars_list.index(v)
                A[i, col] += coeff

        return A, b
    import sympy as sp

    lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8 = sp.symbols('lambda1:9')
    q, p = sp.symbols('q p')
    kappa, gamma, g0, Re_eta, Im_eta = sp.symbols('kappa gamma g0 Re_eta Im_eta')
    Gamma, Delta1, Delta2, Omega, V = sp.symbols('Gamma Delta1 Delta2 Omega V')

    # --- Define the RHS of each equation f1...f10 ---
    f1 = -kappa/2 * q + gamma*g0/sp.sqrt(2)*lambda2 + sp.sqrt(2)*Re_eta
    f2 = -kappa/2 * p - gamma*g0/sp.sqrt(2)*lambda1 + sp.sqrt(2)*Im_eta

    f3 = -Gamma/2 * lambda1 \
        + Delta1 * lambda2 \
        + Omega/2 * lambda5 \
        + sp.sqrt(2)*gamma*g0*lambda3*p

    f4 = -Gamma/2 * lambda2 \
        - Delta1 * lambda1 \
        - Omega/2 * lambda4 \
        - sp.sqrt(2)*gamma*g0*lambda3*q

    f5 = -Gamma * lambda3 \
        + (sp.sqrt(3)*Gamma/3) * lambda8 \
        + 2*Gamma/3 \
        - Omega/2 * lambda7 \
        - sp.sqrt(2)*gamma*g0*(lambda1*p - lambda2*q)

    f6 = sp.I * (
            Delta2*lambda4
            + Omega/2*lambda1
            - 2*sp.sqrt(3)/3*V*lambda4*lambda8
            - 2*V/3*(lambda4**2 - lambda5**2 + 2*lambda7**2)
            + sp.sqrt(2)/2*gamma*g0*(lambda7*p - lambda6*q)
        )

    f7 = sp.I * (
            Delta2*lambda5
            - Omega/2*lambda2
            - 2*sp.sqrt(3)/3*V*lambda5*lambda8
            - 2*V/3*(lambda5**2 - lambda4**2 + 2*lambda6**2)
            + sp.sqrt(2)/2*gamma*g0*(lambda6*p + lambda7*q)
        )

    f8 = -Gamma/2 * lambda6 \
        + (Delta2 - Delta1)*lambda7 \
        + Omega/2*lambda4 \
        - 2*sp.sqrt(3)/3*V*lambda6*lambda8 \
        - 2*V/3*(lambda4*lambda7 - lambda5*lambda6) \
        + sp.sqrt(2)/2*gamma*g0*(lambda4*p - lambda5*q)

    f9 = -Gamma/2 * lambda7 \
        - (Delta2 - Delta1)*lambda6 \
        - Omega/2*lambda5 \
        - 2*sp.sqrt(3)/3*V*lambda7*lambda8 \
        - 2*V/3*(lambda4*lambda6 + lambda5*lambda7) \
        + sp.sqrt(2)/2*gamma*g0*(lambda4*q + lambda5*p)

    f10 = sp.sqrt(3)/2 * Omega * lambda7

    # --- Assemble state vector in order x = [λ1...λ8, q, p] ---
    x = sp.Matrix([
        lambda1, lambda2, lambda3, lambda4,
        lambda5, lambda6, lambda7, lambda8,
        q, p
    ])

    # --- Assemble vector field in the same order ---
    f = sp.Matrix([f3, f4, f5, f6, f7, f8, f9, f10, f1, f2])

    # # --- Compute Jacobian D(x) = ∂f/∂x ---
    # D = f.jacobian(x)
    # D_simplified = sp.simplify(D)
    D_full = f.jacobian(x)      # x = [λ1 … λ8, q, p]

    # copy to keep the full version if you need it later
    D_clean = D_full.copy()

    D_clean[:8, 8:10] = sp.zeros(8, 2)     # q- und p-Spalten (Index 8,9) der ersten 8 Zeilen

    # Variant B – einzeln pro Element (läuft immer, auch in älteren SymPy-Versionen)
    for i in range(8):                     # only λ₁ … λ₈ rows
        D_clean[i, 8] = 0                  # q-column
        D_clean[i, 9] = 0                  # p-column
        # now simplify – duplicates are gone
    D=    D_clean
    D_simplified = sp.simplify(D_clean)
    lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8 = sp.symbols('lambda1:9')
    lambdas = sp.Matrix([lambda1, lambda2, lambda3, lambda4,
                     lambda5, lambda6, lambda7, lambda8])
    F = sp.Matrix([f3, f4, f5, f6, f7, f8, f9, f10])

    F  = sp.Matrix([f3, f4, f5, f6, f7, f8, f9, f10, f1, f2])
    var = sp.Matrix([lambda1, lambda2, lambda3, lambda4,
                    lambda5, lambda6, lambda7, lambda8, q, p])

    # 5) Hilfsroutine:   build A,b  ohne Doppelspalten
    #    (gleiche Logik wie zuvor – jeder Nichtlinearitätsterm
    #     kommt NUR in die Spalte mit dem kleinsten Index)
    # ---------------------------------------------------------------------
    def single_column_matrix(eq_list, vars_list, rule='min'):
        """Return (A, b) with every nonlinear term assigned to ONE column."""
        n, m = len(eq_list), len(vars_list)
        A = sp.zeros(n, m)
        b = sp.zeros(n, 1)

        for i, expr in enumerate(eq_list):
            expr = sp.expand(expr)
            for term in expr.as_ordered_terms():
                involved = [v for v in vars_list if term.has(v)]
                if not involved:
                    b[i] += term
                    continue
                v = min(involved, key=vars_list.index) if rule == 'min' \
                    else max(involved, key=vars_list.index)
                col = vars_list.index(v)
                A[i, col] += sp.expand(term / v)
        return A, b

    A, B = single_column_matrix(list(F), list(var), rule='min')
    D_simplified = sp.simplify(A)
    B = sp.simplify(B)

    sp.pprint(B)


    # --- Print results (console output auf Deutsch) ---
    #print("Jacobi-Matrix D(x):")
    #sp.pprint(D_simplified)

    #print("\nInhomogener Vektor b(x):")
    #sp.pprint(b)
    def get_blocks(D_simplified):
        """
        Given a 10x10 time-dependent matrix D,
        extract the four blocks corresponding to
        system (8x8), system→bath (8x2), bath→system (2x8),
        and bath (2x2).
        """
        # system-system block (8×8)
        D_SS = D[:8, :8]
        # system-bath block (8×2)
        D_SB = D[:8, 8:]
        # bath-system block (2×8)
        D_BS = D[8:, :8]
        # bath-bath block (2×2)
        D_BB = D[8:, 8:]
        return D_SS, D_SB, D_BS, D_BB

    D_SS, D_SB, D_BS, D_BB = get_blocks(D_simplified)

    #sp.pprint(D_SS)
    initial_state=[0,0,1,0,0,0,0,1/np.sqrt(3),0,0]  # state sigma_00=1
    #initial_state=[0,0,-1,0,0,0,0,1/np.sqrt(3),0,0]  # state sigma_11=1
    #initial_state=[0,0,0,0,0,0,0,-2/np.sqrt(3),0,0]  # state sigma_22=1
    s0,s_s0,s_b0=symplect.get_symplectic(initial_state)
    st,s_st,s_bt=symplect.get_symplectic()

    comutator_mDs=-D_SS*s_st+s_st*D_SS
    #sp.pprint(st)

    theta = sp.symbols('theta', real=True)
    cos_theta = sp.cos(theta)
    sin_theta = sp.sin(theta)
    from sympy import symbols, Matrix, sin, cos, eye, zeros
    # --- 1) Symbole definieren ---
    theta = symbols('theta')
    lambda_syms = symbols('lambda1:9')  # λ1…λ8
    q_sym, p_sym = symbols('q p', commutative=False)

    # symbolic angle
    phi = sp.symbols('phi', real=True)

    # convenience shorthands
    c = sp.cos(phi/2)
    s = sp.sin(phi/2)

    # 8×8 identity
    R0 = sp.eye(8)

    # rotate (T1,T2)
    R0[0,0], R0[0,1], R0[1,0], R0[1,1] = c, -s, s, c

    # rotate (T4,T5)
    R0[3,3], R0[3,4], R0[4,3], R0[4,4] = c, -s, s, c


    zero_8x2 = zeros(8, 2)
    zero_2x8 = zeros(2, 8)
    I2 = sp.eye(2)
    R = R0.row_join(zero_8x2).col_join(zero_2x8.row_join(I2))

    # --- 3) Symbolische s(0) holen (nur SU(3)-Block C und Quadraturen b) ---
    # Angenommen Du hast eine Funktion get_symplectic(), die bei v=None S_s symbolisch liefert:
    # S_s, C_s, b_s = get_symplectic()
    # Hier bauen wir S_s manuell (analog zur Funktion):
    # SU(3)-Strukturkonstanten usw. übersprungen – wir nehmen an, S_sym ist da:
    S_sym, _, _ = symplect.get_symplectic()

    # --- 4) check_s0 berechnen ---
    check_s0 = R.T * S_sym * R

    # --- 5) Werte einsetzen ---
    check_state = {
        lambda_syms[i]: initial_state[i] for i in range(8)
    }
    #[0,0,1,0,0,0,0,1/np.sqrt(3),0,0] 
    #initial_state[lambda_syms[7]] = 1/sp.sqrt(3)
    #initial_state[lambda_syms[2]] = 1


    #initial_state=[0,0,0,0,0,0,0,-2/np.sqrt(3),0,0]  # state sigma_22=1
    #initial_state[lambda_syms[7]] = -2/sp.sqrt(3)
    check_state[q_sym] = 0
    check_state[p_sym] = 0

    check_s0_sub = sp.simplify(check_s0.subs(check_state))

    #sp.pprint(R)
    #sp.pprint(check_s0)
    #sp.pprint(s0)
    #sp.pprint(check_s0_sub)




    dRtsRdt=R0.T*comutator_mDs*R0#+R.T*dsdt*R
    return dRtsRdt, R0,R,st,s_st,s_bt,s0,s_s0,s_b0,D_SS, D_SB, D_BS, D_BB,D_simplified,B,dxdt



# if __name__=="__main__":
#     create_all()