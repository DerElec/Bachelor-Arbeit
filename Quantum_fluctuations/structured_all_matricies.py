import sympy as sp
import numpy as np
def run_all():
    # -----------------------------
    # Symbolische Konstanten
    # -----------------------------
    Gamma, kappa = sp.symbols('Gamma kappa')
    m = sp.symbols('m1:9')  # m1, ..., m8
    I = sp.I

    # Gell-Mann Matrizen (sympy)
    lambdas = [
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        sp.Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
        sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        sp.Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]),
        sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) * (1/sp.sqrt(3))
    ]

    def build_gellmann():
        I = sp.I
        sqrt3 = sp.sqrt(3)

        lambdas = [
            sp.Matrix([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]]),
            sp.Matrix([[0, -I, 0],
                    [I,  0, 0],
                    [0,  0, 0]]),
            sp.Matrix([[1,  0, 0],
                    [0, -1, 0],
                    [0,  0, 0]]),
            sp.Matrix([[0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0]]),
            sp.Matrix([[0,  0, -I],
                    [0,  0,  0],
                    [I,  0,  0]]),
            sp.Matrix([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]]),
            sp.Matrix([[0,  0,  0],
                    [0,  0, -I],
                    [0,  I,  0]]),
            (1/sqrt3)*sp.Matrix([[ 1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0,-2]])
        ]
        return lambdas


    def structure_constants_from_lambdas(lambdas):
        """
        Compute SU(3) structure constants f^{abc} and d^{abc} from the Gell-Mann matrices.

        Conventions:
        [λ_a, λ_b] = 2i f^{abc} λ_c
        {λ_a, λ_b} = (4/3) δ_{ab} I + 2 d^{abc} λ_c
        Trace formulas used:
        f^{abc} = (1 / (4 i)) tr([λ_a, λ_b] λ_c)
        d^{abc} = (1 / 4)     tr({λ_a, λ_b} λ_c)
        """
        PHYS_DIM = 8
        I = sp.I

        f = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]
        d = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]

        for a in range(PHYS_DIM):
            for b in range(PHYS_DIM):
                comm   = lambdas[a]*lambdas[b] - lambdas[b]*lambdas[a]
                anticom= lambdas[a]*lambdas[b] + lambdas[b]*lambdas[a]
                for c in range(PHYS_DIM):
                    f_val = sp.simplify(sp.trace(comm * lambdas[c])/(4*I))
                    d_val = sp.simplify(sp.trace(anticom * lambdas[c])/4)
                    f[a][b][c] = f_val
                    d[a][b][c] = d_val
        return f, d


    def compute_R_from_f_d(f_arr, d_arr, Gamma=None, simplify_entries=False):
        """
        Build the 8x8 diffusion matrix R_ab using numeric structure constants f,d
        (nested Python lists with SymPy numbers).

        R_{ab} = γ ∑_{c,d} [ (2/3) δ_{cd} + ∑_{k} m_k ( d^{cdk} + i f^{cdk} ) ]
                            * ( f^{1ac} - i f^{2ac} ) * ( f^{1bd} + i f^{2bd} )
        """
        I = sp.I
        PHYS_DIM = 8
        if Gamma is None:
            Gamma = sp.symbols('Gamma', real=True)

        # IMPORTANT: treat m_k as real so that "realness" checks can simplify
        m = sp.symbols('m1:9', real=True)  # (m1,...,m8) are real

        R = sp.MutableDenseMatrix(PHYS_DIM, PHYS_DIM, [0]*(PHYS_DIM*PHYS_DIM))

        for a in range(1, PHYS_DIM+1):
            for b in range(1, PHYS_DIM+1):
                expr = 0
                for c in range(1, PHYS_DIM+1):
                    U_ac = f_arr[0][a-1][c-1] - I * f_arr[1][a-1][c-1]
                    for d in range(1, PHYS_DIM+1):
                        V_bd = f_arr[0][b-1][d-1] + I * f_arr[1][b-1][d-1]

                        delta_cd = 1 if c == d else 0
                        T_cd = sp.Rational(2, 3)*delta_cd
                        T_cd += sum(m[k-1]*(d_arr[c-1][d-1][k-1] + I*f_arr[c-1][d-1][k-1])
                                    for k in range(1, PHYS_DIM+1))

                        expr += T_cd * U_ac * V_bd

                R[a-1, b-1] = Gamma * (sp.simplify(expr) if simplify_entries else expr)

        return sp.Matrix(R), {'Gamma': Gamma, 'm': m, 'f': f_arr, 'd': d_arr}


    def compute_R_su3(Gamma=None, simplify_entries=False):
        """
        Convenience wrapper:
        - builds Gell-Mann matrices,
        - computes numeric f,d,
        - returns R_ab.

        Returns
        -------
        R : sympy.Matrix (8x8)
        data : dict with 'Gamma','m','f','d','lambdas'
        """
        lambdas = build_gellmann()
        f_arr, d_arr = structure_constants_from_lambdas(lambdas)
        R, syms = compute_R_from_f_d(f_arr, d_arr, Gamma=Gamma, simplify_entries=simplify_entries)
        syms.update({'lambdas': lambdas})
        return R, syms

    # -----------------------------
    # Strukturkonstanten f und d Arrays
    # -----------------------------
    PHYS_DIM = 8

    def build_structure_arrays():
        f = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]
        d = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]
        for a in range(PHYS_DIM):
            for b in range(PHYS_DIM):
                comm = lambdas[a]*lambdas[b] - lambdas[b]*lambdas[a]
                anticom = lambdas[a]*lambdas[b] + lambdas[b]*lambdas[a]
                for c in range(PHYS_DIM):
                    f_val = sp.trace(comm * lambdas[c])/(4*sp.I)
                    d_val = sp.trace(anticom * lambdas[c])/4
                    f[a][b][c] = sp.simplify(f_val)
                    d[a][b][c] = sp.simplify(d_val)
        return f, d

    f_arr, d_arr = build_structure_arrays()

    # Hilfsfunktion für f
    f_sym = lambda a,b,c: f_arr[a-1][b-1][c-1] if 1<=a<=8 and 1<=b<=8 and 1<=c<=8 else 0


    def check_half_MM_T_real(M):
        """
        Return (ok, ImPart) where ok=True iff (M*M.T)/2 has zero imaginary part.
        """
        S = (M * M.T) / 2
        ImS = S.applyfunc(lambda x: sp.simplify(sp.im(x)))
        ok = all(el == 0 for el in ImS)  # element-wise test
        return ok, ImS
    






    
    # -----------------------------
    # sDs-Matrix Berechnung
    # -----------------------------
    def compute_sDs():
        M = sp.zeros(10)
        p,q = 10,9
        for a in range(1,11):
            for c in range(1,11):
                M[a-1,c-1] = sp.simplify(kappa/2*(
                    sp.KroneckerDelta(a,p)*sp.KroneckerDelta(c,p)
                    + sp.KroneckerDelta(a,q)*sp.KroneckerDelta(c,q)
                ))
        return M

    sDs = compute_sDs()

    # -----------------------------
    # sE-Matrix Berechnung
    # -----------------------------
    def compute_sE():
        E = sp.zeros(10)
        E[8,8] = -sp.Rational(1,2)
        E[9,9] = -sp.Rational(1,2)
        return E

    sE = compute_sE()

    # -----------------------------
    # P-Matrix Berechnung
    # -----------------------------


    # --- Symbols (keep same names as in your code) ---
    Delta1, Delta2, Omega = sp.symbols('Delta1 Delta2 Omega', real=True)
    V_const, eta, g0   = sp.symbols('V eta g0', real=True)
    # Singles m1..m10 (1..8 = SU(3); 9 = q; 10 = p)
    mP = sp.symbols('m1:11')

    # You must provide f_sym(i,a,G) elsewhere (unchanged).
    def f_P(i, a, G):
        """Return f_{i a G} for SU(3) part, else 0 (keeps your old contract)."""
        if 1 <= i <= 8 and 1 <= a <= 8 and 1 <= G <= 8:
            return f_sym(i, a, G)
        return 0

    # --- Linear coefficients ω_i from corrected H_N ---
    # ω_3 = -Δ1/2 ; ω_6 = +Ω/2 ; ω_8 = (Δ1/2 - Δ2 - 2V/3)/√3 ; ω_10 = 2√N η
    omega = {i: sp.Integer(0) for i in range(1, 11)}
    omega[3]  = -Delta1/2
    omega[6]  =  Omega/2
    omega[8]  = (Delta1/2 - Delta2 - 2*V_const/3) / sp.sqrt(3)
    omega[10] =  sp.sqrt(2)*eta  # belasse √N, wie du es vorher genutzt hast

    # --- Quadratic couplings h_{μν} (symmetric), with γ absorbed in g0 ---
    # Convention: H_ia = (ħ/N) * sum_{μν} h_{μν} m_μ m_ν
    h = {}
    def set_h(mu, nu, val):
        h[(mu, nu)] = val
        h[(nu, mu)] = val

    # Light–matter: g0 (m1*q + m2*p)  -> note: plus for m2*p with your q,p def
    set_h(1, 9,  g0/sp.sqrt(2))
    set_h(2, 10, -g0/sp.sqrt(2))

    # V-term quadratic piece: (V/3) m8^2
    set_h(8, 8, V_const/3)

    # --- Build symbolic P(m; params) ---
    def build_P_sym():
        P = sp.zeros(10, 10)

        # (1) Linear H0 contribution: P^{(0)}_{aG} = sum_i ω_i f_{i a G}
        for a in range(1, 11):
            for G in range(1, 11):
                if a <= 8 and G <= 8:
                    P[a-1, G-1] += sum(2*omega[i] * f_P(i, a, G) for i in (3, 6, 8)) # ä#ääääääääääääääääääääääääääääänderung   <----------------------------
                # (bosonic rows a=9,10: no f-contribution from H0)

        # (2) Bilinear H_ia contribution:
        #     P^{(ia)}_{aG} = (2/N) * sum_{μ,ν} h_{μν} * m_μ * f_{ν a G}    (ν<=8)
        #                   + (2/N) * sum_{ν} h_{Gν} * sum_c f_{ν a c} m_c  (only for G=9,10)
        for a in range(1, 11):
            for G in range(1, 11):
                # First term: atomic columns G<=8
                if a <= 8 and G <= 8:
                    s = 0
                    for (mu, nu), hval in h.items():
                        if 1 <= nu <= 8:
                            s += hval * mP[mu-1] * f_P(nu, a, G)  #ääääääääääääääääääääääääääääänderung  <----------------------------
                    P[a-1, G-1] += 2 * s

                # Second term: bosonic columns (G=9,10)
                if a <= 8 and G in (9, 10):
                    s = 0
                    for nu in range(1, 9):
                        hGnu = h.get((G, nu), 0)
                        if hGnu != 0:
                            s += hGnu * sum(f_P(nu, a, c) * mP[c-1] for c in range(1, 9))
                    if s != 0:
                        P[a-1, G-1] += 2 * s

        # (3) Bosonic rows from canonical equations:
        #     qdot = + g0 * m2 + 2N*eta     -> P[9,2]  = + g0  (inhom. 2N*eta not in P)
        #     pdot = - g0 * m1              -> P[10,1] = - g0
        P[9-1,  2-1] +=  g0
        P[10-1, 1-1] += -g0

        return sp.simplify(P)

   

    P=build_P_sym()
    P = sp.simplify(P)
    #sp.pprint(P)
    print("P-Matrix (symbolisch) aufgebaut.")
    # -----------------------------
    # Q-Matrix Berechnung
    # -----------------------------
    DIM = 10

    def compute_Q():
        Q = sp.zeros(DIM)
        for alpha in range(1,PHYS_DIM+1):
            for k in range(1,PHYS_DIM+1):
                tot = 0
                for c in range(1,PHYS_DIM+1):
                    f_a1c = f_sym(alpha,1,c); f_a2c = f_sym(alpha,2,c)
                    f_1ac = f_sym(1,alpha,c); f_2ac = f_sym(2,alpha,c)
                    d_1ck = d_arr[0][c-1][k-1]; f_1ck = f_sym(1,c,k)
                    d_2ck = d_arr[1][c-1][k-1]; f_2ck = f_sym(2,c,k)
                    d_c1k = d_arr[c-1][0][k-1]; f_c1k = f_sym(c,1,k)
                    d_c2k = d_arr[c-1][1][k-1]; f_c2k = f_sym(c,2,k)
                    t1 = I*f_a1c*(d_1ck + I*f_1ck)
                    t2 =     f_a1c*(d_2ck + I*f_2ck)
                    t3 = -   f_a2c*(d_1ck + I*f_1ck)
                    t4 = I*f_a2c*(d_2ck + I*f_2ck)
                    t5 = I*f_1ac*(d_c1k + I*f_c1k)
                    t6 =   - f_1ac*(d_c2k + I*f_c2k)
                    t7 =     f_2ac*(d_c1k + I*f_c1k)
                    t8 = I*f_2ac*(d_c2k + I*f_c2k)
                    tot += t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
                Q[k-1,alpha-1] = sp.simplify(tot)
        return (Gamma/4)*Q

    Q = compute_Q()

    # -----------------------------
    # Matrix G = P + sE + Q
    # -----------------------------
    sE = kappa * compute_sE()
    def compute_Z_prime_new():
        """
        NEW Z' with conjugate pairing so that Im((Z'+Z'.T)/2) cancels.
        Z'_{ab} = γ Σ_{c,d} [ (2/3) δ_cd + Σ_k m_k ( d^{cdk} + i f^{cdk} ) ] * (f^{1ac} - i f^{2ac}) * (f^{1bd} + i f^{2bd})
        Embed 8x8 block into 10x10 (top-left).
        """
        I = sp.I
        PHYS_DIM = 8
        DIM = 10

        Z8 = sp.MutableDenseMatrix(PHYS_DIM, PHYS_DIM, [0]*(PHYS_DIM*PHYS_DIM))

        for a in range(1, PHYS_DIM+1):
            for b in range(1, PHYS_DIM+1):
                expr = 0
                for c in range(1, PHYS_DIM+1):
                    # f_ac with minus i  (conjugate partner)
                    f_ac_minus = f_sym(1, a, c) - I * f_sym(2, a, c)
                    for d in range(1, PHYS_DIM+1):
                        # f_bd with plus i
                        f_bd_plus = f_sym(1, b, d) + I * f_sym(2, b, d)

                        # T_cd = (2/3) δ_cd + sum_k m_k ( d^{cdk} + i f^{cdk} )
                        delta_cd = 1 if c == d else 0
                        T_cd = sp.Rational(2, 3) * delta_cd
                        T_cd += sum(
                            m[k-1] * ( d_arr[c-1][d-1][k-1] + I * f_arr[c-1][d-1][k-1] )
                            for k in range(1, PHYS_DIM+1)
                        )

                        expr += T_cd * f_ac_minus * f_bd_plus

                Z8[a-1, b-1] = Gamma * sp.simplify(expr)

        # Embed 8x8 into 10x10 (top-left)
        Zp = sp.zeros(DIM)
        for i in range(PHYS_DIM):
            for j in range(PHYS_DIM):
                Zp[i, j] = Z8[i, j]
        return Zp


    G = P + sE + Q#P + sE + Q

    Z_prime=compute_Z_prime_new()
    Z=sp.simplify((Z_prime+Z_prime.T)/2)
    W=Z+sDs
    print("done computing Z, Z', sDs, Q, G ...")
    return G,sDs,Z,P,Q,sE,Z_prime,W

