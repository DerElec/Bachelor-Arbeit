# -*- coding: utf-8 -*-
# ==============================================
#  Translation of the provided SymPy‑based Python
#  code into Julia using the SymPy.jl wrapper.
#
#  Author: ChatGPT (OpenAI), 2025‑07‑02
#  Purpose: Calculate the matrices Z, Z', sDs, P, Q
#           and the composite matrix G for SU(3)
#           algebraic structures.
#
#  Notes
#  -----
#  • Requires Julia ≥ 1.9 and the package SymPy.jl.
#    Install with: ] add SymPy
#  • All comments are in English as requested by Paul.
#  • Console output remains German.
# ==============================================

using SymPy
@syms gamma kappa
m = [symbols("m$i") for i in 1:8]
I = SymPy.I

inv√3 = Sym(1) / sqrt(Sym(3))

lambdas = [
    SymPy.Matrix([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]]),
    SymPy.Matrix([[0, -I, 0],
                  [I,   0, 0],
                  [0,   0, 0]]),
    SymPy.Matrix([[1, 0, 0],
                  [0,-1, 0],
                  [0, 0, 0]]),
    SymPy.Matrix([[0, 0, 1],
                  [0, 0, 0],
                  [1, 0, 0]]),
    SymPy.Matrix([[0, 0, -I],
                  [0, 0,  0],
                  [I, 0,  0]]),
    SymPy.Matrix([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]]),
    SymPy.Matrix([[0, 0,  0],
                  [0, 0, -I],
                  [0, I,  0]]),
    inv√3 * SymPy.Matrix([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0,-2]])
]

# -----------------------------
# Structure constants f and d
# -----------------------------
PHYS_DIM = 8                     # dimension of SU(3) adjoint

f_arr = zeros(Sym,  PHYS_DIM, PHYS_DIM, PHYS_DIM)
d_arr = zeros(Sym,  PHYS_DIM, PHYS_DIM, PHYS_DIM)

for a in 1:PHYS_DIM, b in 1:PHYS_DIM
    comm    = lambdas[a] * lambdas[b] - lambdas[b] * lambdas[a]
    anticom = lambdas[a] * lambdas[b] + lambdas[b] * lambdas[a]
    for c in 1:PHYS_DIM
        f_arr[a,b,c] = simplify(trace(comm    * lambdas[c]) / (4*I))
        d_arr[a,b,c] = simplify(trace(anticom * lambdas[c]) / 4)
    end
end

# Helper returning f^{abc} with 1‑based indices (returns 0 out of range)
f_sym(a,b,c) = (1 ≤ a ≤ 8 && 1 ≤ b ≤ 8 && 1 ≤ c ≤ 8) ? f_arr[a,b,c] : 0

# -----------------------------
# Z'‑matrix
# -----------------------------
Zp = zeros(Sym, 10, 10)
for a in 1:8, b in 1:8
    Zp[a,b] = simplify(sum(
        gamma * m[c] * m[d] * (
            f_sym(1,a,c+1) * f_sym(b,1,d+1) +
            I * f_sym(1,a,c+1) * f_sym(b,2,d+1) -
            I * f_sym(2,a,c+1) * f_sym(b,1,d+1) +
            f_sym(2,a,c+1) * f_sym(b,2,d+1)
        )
        for c in 0:7, d in 0:7))
end

Z_temp = (Zp + transpose(Zp))/2
Z      = SymPy.Matrix(10, 10, (i,j)->simplify(expand(Z_temp[i,j])))

# -----------------------------
# sDs‑matrix
# -----------------------------
sDs = zeros(Sym, 10, 10)
p, q = 10, 9                       # special indices
for a in 1:10, c in 1:10
    sDs[a,c] = simplify(kappa/8 * (
        SymPy.KroneckerDelta(a,p) * SymPy.KroneckerDelta(c,p) +
        SymPy.KroneckerDelta(a,q) * SymPy.KroneckerDelta(c,q)
    ))
end

# -----------------------------
# sE‑matrix (un‑scaled)
# -----------------------------
function compute_sE()
    E = zeros(Sym, 10, 10)
    E[9,9]  = -Sym(1)//2
    E[10,10] = -Sym(1)//2
    return E
end
sE = compute_sE()

# -----------------------------
# P‑matrix
# -----------------------------
mP = [symbols("m$i") for i in 1:10]

g0, V_const = symbols("g0 V")
Delta1, Omega, N, eta = symbols("Delta1 Omega N eta")

h9  = g0
h10 = -g0
h88 = V_const/3
omega = [0, 0, -Delta1, 0, 0, Omega, 0, -V_const/(3*sqrt(3)), 0, 2*sqrt(N)*eta]

# Helper that returns f for allowed index range only
f_P(g,a,c) = (g ≤ 8 && a ≤ 8 && c ≤ 8) ? f_sym(g,a,c) : 0

P = zeros(Sym, 10, 10)
for a in 1:10, G in 1:10
    coeff = Sym(0)
    if G == 9
        sumA = sum(f_P(1,a,c) * mP[c] for c in 1:8)
        coeff -= h9 * 2 * sumA
    end
    if G ≤ 8
        coeff -= h9 * 2 * mP[9] * f_P(1,a,G)
    end
    if G == 1
        coeff -= h9 * (a == 10 ? Sym(1)//2 : 0)
    end
    if G == 10
        sumB = sum(f_P(2,a,c) * mP[c] for c in 1:8)
        coeff -= h10 * 2 * sumB
    end
    if G ≤ 8
        coeff -= h10 * 2 * mP[10] * f_P(2,a,G)
    end
    if G == 2
        coeff += h10 * (a == 9 ? Sym(1)//2 : 0)
    end
    if G == 8
        coeff -= h88 * 4 * f_P(8,a,G) * mP[G]
    end
    if G ≤ 8
        coeff -= h88 * 4 * mP[8] * f_P(8,a,G)
    end
    if G ≤ 8
        for i in 1:8
            coeff -= 2 * omega[i] * f_P(i,a,G)
        end
    end
    P[a,G] = factor(coeff)
end

# -----------------------------
# Q‑matrix
# -----------------------------
Q = zeros(Sym, 10, 10)
for alpha in 1:PHYS_DIM, k in 1:PHYS_DIM
    tot = Sym(0)
    for c in 1:PHYS_DIM
        f_a1c = f_sym(alpha,1,c)
        f_a2c = f_sym(alpha,2,c)
        f_1ac = f_sym(1,alpha,c)
        f_2ac = f_sym(2,alpha,c)
        d_1ck = d_arr[1,c,k]
        f_1ck = f_sym(1,c,k)
        d_2ck = d_arr[2,c,k]
        f_2ck = f_sym(2,c,k)
        d_c1k = d_arr[c,1,k]
        f_c1k = f_sym(c,1,k)
        d_c2k = d_arr[c,2,k]
        f_c2k = f_sym(c,2,k)

        # 8 individual terms
        t1 =  I * f_a1c * (d_1ck + I * f_1ck)
        t2 =      f_a1c * (d_2ck + I * f_2ck)
        t3 =  -   f_a2c * (d_1ck + I * f_1ck)
        t4 =  I * f_a2c * (d_2ck + I * f_2ck)
        t5 =  I * f_1ac * (d_c1k + I * f_c1k)
        t6 = -    f_1ac * (d_c2k + I * f_c2k)
        t7 =      f_2ac * (d_c1k + I * f_c1k)
        t8 =  I * f_2ac * (d_c2k + I * f_c2k)

        tot += t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    end
    Q[k,alpha] = simplify(tot)
end
Q .*= gamma / 4

# -----------------------------
# Final composite matrix G
# -----------------------------
sE_scaled = kappa * sE
G = P + sE_scaled + Q

println("Fertig mit Z, Z', sDs, Q, G …")
return G, sDs, Z, P, Q, Zp
