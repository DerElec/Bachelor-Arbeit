import sympy as sp
def run_all():
    # -----------------------------
    # Symbolische Konstanten
    # -----------------------------
    gamma, kappa = sp.symbols('gamma kappa')
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


    def compute_R_from_f_d(f_arr, d_arr, gamma=None, simplify_entries=False):
        """
        Build the 8x8 diffusion matrix R_ab using numeric structure constants f,d
        (nested Python lists with SymPy numbers).

        R_{ab} = γ ∑_{c,d} [ (2/3) δ_{cd} + ∑_{k} m_k ( d^{cdk} + i f^{cdk} ) ]
                            * ( f^{1ac} - i f^{2ac} ) * ( f^{1bd} + i f^{2bd} )
        """
        I = sp.I
        PHYS_DIM = 8
        if gamma is None:
            gamma = sp.symbols('gamma', real=True)

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

                R[a-1, b-1] = gamma * (sp.simplify(expr) if simplify_entries else expr)

        return sp.Matrix(R), {'gamma': gamma, 'm': m, 'f': f_arr, 'd': d_arr}


    def compute_R_su3(gamma=None, simplify_entries=False):
        """
        Convenience wrapper:
        - builds Gell-Mann matrices,
        - computes numeric f,d,
        - returns R_ab.

        Returns
        -------
        R : sympy.Matrix (8x8)
        data : dict with 'gamma','m','f','d','lambdas'
        """
        lambdas = build_gellmann()
        f_arr, d_arr = structure_constants_from_lambdas(lambdas)
        R, syms = compute_R_from_f_d(f_arr, d_arr, gamma=gamma, simplify_entries=simplify_entries)
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

    # -----------------------------
    # Z- und Z'-Matrix Berechnung
    # -----------------------------
    # def compute_Z_prime():
    #     Zp = sp.zeros(10)
    #     for a in range(1,9):
    #         for b in range(1,9):
    #             expr = sum(
    #                 gamma * m[c-1] * m[d-1] * (
    #                     f_sym(1,a,c)*f_sym(b,1,d)
    #                     + I*f_sym(1,a,c)*f_sym(b,2,d)
    #                     - I*f_sym(2,a,c)*f_sym(b,1,d)
    #                     + f_sym(2,a,c)*f_sym(b,2,d)
    #                 )
    #                 for c in range(1,9) for d in range(1,9)
    #             )
    #             Zp[a-1,b-1] = sp.simplify(expr)
    #     return Zp

    # Z_prime = compute_Z_prime()
    # Z_temp = (Z_prime + Z_prime.T)/2
    # Z = sp.Matrix(10, 10, lambda i,j: sp.simplify(sp.expand(Z_temp[i,j])))
    def check_half_MM_T_real(M):
        """
        Return (ok, ImPart) where ok=True iff (M*M.T)/2 has zero imaginary part.
        """
        S = (M * M.T) / 2
        ImS = S.applyfunc(lambda x: sp.simplify(sp.im(x)))
        ok = all(el == 0 for el in ImS)  # element-wise test
        return ok, ImS
    

    def compute_Z_prime():
        """
        1) Build 8x8 matrix Z8 (a.k.a. R_ab) from SU(3) structure constants.
        2) Check if (Z8 * Z8.T)/2 is real.
        3) Embed Z8 into a 10x10 matrix (top-left block), zeros elsewhere.
        """
        Z8, syms = compute_R_su3(gamma=None, simplify_entries=False)

        # Realness check for (Z8 * Z8^T) / 2
        ok, ImS = check_half_MM_T_real(Z8)
        print(f"Check: (Z' * Z'^T)/2 ist reell? {ok}")
        if not ok:
            print("Nicht verschwindender Imaginärteil von (Z' * Z'^T)/2 (symbolisch):")
            sp.pprint(ImS)

        # Embed into 10x10 (top-left block)
        Zp = sp.zeros(10)
        for i in range(8):
            for j in range(8):
                Zp[i, j] = Z8[i, j]
        return Zp
    






    
    # -----------------------------
    # sDs-Matrix Berechnung
    # -----------------------------
    def compute_sDs():
        M = sp.zeros(10)
        p,q = 10,9
        for a in range(1,11):
            for c in range(1,11):
                M[a-1,c-1] = sp.simplify(kappa/8*(
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
    mP = sp.symbols('m1:11')
    #F = sp.symbols('F1:11')
    g0, V_const = sp.symbols('g0 V')
    Delta1,Delta2, Omega, N, eta = sp.symbols('Delta1 Delta2 Omega N eta')

    h9 = g0; h10 = -g0; h88 = V_const/3
    omega = [0,0,-Delta1,0,0,Omega,0,-2*V_const/(3*sp.sqrt(3))-(2/sp.sqrt(3))*Delta2,0,2*sp.sqrt(N)*eta]

    f_P = lambda g,a,c: f_sym(g,a,c) if g<=8 and a<=8 and c<=8 else 0

    P = sp.zeros(10)
    for a in range(1,11):
        for G in range(1,11):
            coeff = 0
            if G==9:
                sumA = sum(f_P(1,a,c)*mP[c-1] for c in range(1,9))
                coeff += -h9*2*sumA
            if G<=8:
                coeff += -h9*2*mP[8]*f_P(1,a,G)
            if G==1:
                coeff += -h9*(sp.Rational(1,2) if a==10 else 0)
            if G==10:
                sumB = sum(f_P(2,a,c)*mP[c-1] for c in range(1,9))
                coeff += -h10*2*sumB
            if G<=8:
                coeff += -h10*2*mP[9]*f_P(2,a,G)
            if G==2:
                coeff += h10*(sp.Rational(1,2) if a==9 else 0)
            if G==8:
                coeff += -h88*4*f_P(8,a,G)*mP[G-1]
            if G<=8:
                coeff += -h88*4*mP[7]*f_P(8,a,G)
            if G<=8:
                for i in range(1,9): coeff += -omega[i-1]*f_P(i,a,G)*2# - ACHTUNG HIER BEARBEITET
            P[a-1,G-1] = sp.factor(coeff)#*F[G-1]

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
        return (gamma/4)*Q

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

                Z8[a-1, b-1] = gamma * sp.simplify(expr)

        # Embed 8x8 into 10x10 (top-left)
        Zp = sp.zeros(DIM)
        for i in range(PHYS_DIM):
            for j in range(PHYS_DIM):
                Zp[i, j] = Z8[i, j]
        return Zp


    G = P + sE + Q

    Z_prime=compute_Z_prime_new()
    Z=sp.simplify((Z_prime+Z_prime.T)/2)
    W=sDs+Z
    print("done computing Z, Z', sDs, Q, G ...")
    return G,sDs,Z,P,Q,Z_prime,W

