from sympy import symbols, I, Matrix, zeros, sqrt, simplify
from itertools import permutations

def get_symplectic(v=None):
    """
    Wenn v angegeben und Länge 10:
      v[0:8] → Werte für lambda1...lambda8
      v[8]   → Wert für q
      v[9]   → Wert für p
      Dann werden sqrt-Ausdrücke zu numerischen Werten ausgewertet.
    Wenn v=None, liefert nur die symbolischen Matrizen zurück.
    Gibt zurück: (S, C, b)
    """
    # --- 1) Symbole definieren ---
    lambda_syms = symbols('lambda1:9')      # λ₁…λ₈
    q_sym, p_sym = symbols('q p', commutative=False)

    # --- 2) SU(3)-Strukturkonstanten (unabhängige Einträge) ---
    f = {
        (0,1,2): 1,
        (0,3,6):  1/2,
        (1,3,5):  1/2,
        (1,4,6): -1/2,
        (2,3,4):  1/2,
        (2,5,6):  1/2,
        (3,4,7):  sqrt(3)/2,
        (5,6,7): -sqrt(3)/2,
    }
    # Voll antisymmetrisch machen
    f_full = {}
    for (i,j,k), val in f.items():
        for perm in set(permutations((i,j,k),3)):
            # Tauschanzahl für das Vorzeichen
            swaps = 0
            temp_copy = [i,j,k]
            for idx in range(3):
                if temp_copy[idx] != perm[idx]:
                    swap_idx = temp_copy.index(perm[idx])
                    temp_copy[idx], temp_copy[swap_idx] = temp_copy[swap_idx], temp_copy[idx]
                    swaps += 1
            sign = -1 if swaps % 2 else 1
            f_full[perm] = sign * val

    # --- 3) Symbolische 8×8-Kommutator-Matrix C_s aufbauen ---
    C_s = Matrix(8, 8, lambda i, j: 
                 simplify(sum(2 * I * f_full.get((i, j, k), 0) * lambda_syms[k]
                              for k in range(8))))

    # --- 4) Symbolischer Quadraturen-Block b_s ---
    comm_qp = q_sym*p_sym - p_sym*q_sym
    b_s = Matrix([[0,        comm_qp],
                  [-comm_qp, 0      ]])

    # --- 5) Symbolische 10×10-Matrix S_s zusammenfügen ---
    zero_8x2 = zeros(8, 2)
    zero_2x8 = zeros(2, 8)
    top    = C_s.row_join(zero_8x2)
    bottom = zero_2x8.row_join(b_s)
    S_s = top.col_join(bottom)

    # Wenn kein Vektor übergeben, nur symbolisch zurückgeben
    if v is None:
        return S_s, C_s, b_s

    # --- 6) Numerische Substitution ---
    if len(v) != 10:
        raise ValueError("Vektor v muss Länge 10 haben.")
    subs_map = { lambda_syms[i]: v[i] for i in range(8) }
    subs_map.update({ q_sym: v[8], p_sym: v[9] })

    S_num = S_s.subs(subs_map).evalf()  # sqrt-Ausdrücke werden zu floats
    C_num = C_s.subs(subs_map).evalf()
    b_num = b_s.subs(subs_map).evalf()

    # --- 7) Ausgabe auf Deutsch ---
    # print("Matrix s mit eingesetzten Werten (Wurzeln als Zahlen):")
    # print(S_num)
    # print("\nSU(3)-Block C:")
    # print(C_num)
    # print("\nQuadraturen-Block b:")
    # print(b_num)

    return S_num, C_num, b_num

# Beispiel ohne Vektor (symbolisch):
# S_sym, C_sym, b_sym = get_symplectic()

# Beispiel mit Vektor (numerisch):
# values = [1,2,3,4,5,6,7,8, 0.1, 0.2]
# S_val, C_val, b_val = get_symplectic(values)
