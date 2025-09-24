import numpy as np
from sympy import symbols, I, Matrix, zeros, sqrt, simplify, pprint

# (Alle Funktionen bleiben exakt wie im vorherigen Code)

# ==============================================================================
#  FUNKTION 1: ERZEUGUNG DER MATRIX C
# ==============================================================================
def get_C_matrix(v=None):
    # ... (keine Änderung)
    lambda_syms = symbols('lambda1:9')
    f = {
    (0,1,2): 1,
    (0,3,6): 1/2,
    (0,4,5): -1/2,      # <<< fehlte bisher: f_156 = -1/2
    (1,3,5): 1/2,       # f_246 = +1/2 (ist schon korrekt)
    (1,4,6): 1/2,
    (2,3,4): 1/2,
    (2,5,6): -1/2,
    (3,4,7): sqrt(3)/2,
    (5,6,7): sqrt(3)/2,
}
    f_full = {}
    for (i,j,k), val in f.items():
        f_full[(i, j, k)] = f_full[(j, k, i)] = f_full[(k, i, j)] = val
        f_full[(i, k, j)] = f_full[(k, j, i)] = f_full[(j, i, k)] = -val
    C_s = Matrix(8, 8, lambda i, j: simplify(sum(2*I*f_full.get((i,j,k),0)*lambda_syms[k] for k in range(8))))
    if v is None: return C_s
    if len(v) != 8: raise ValueError("Vektor v muss Länge 8 haben.")
    subs_map = {lambda_syms[i]: v[i] for i in range(8)}
    C_num = C_s.subs(subs_map).evalf()
    return C_num

# ==============================================================================
#  HILFSFUNKTION 2: DER INTERNE, REELLE ALGORITHMUS
# ==============================================================================
def _transform_real_skew_matrix(C_real):
    # ... (keine Änderung)
    N = C_real.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(C_real)

    pos_imag_indices = np.where(np.imag(eigenvalues) > 1e-10)[0]
    pos_imag_indices = sorted(pos_imag_indices, key=lambda i: np.imag(eigenvalues[i]), reverse=True)
    selected_eigenvectors = eigenvectors[:, pos_imag_indices]
    num_pos_eigenvalues = selected_eigenvectors.shape[1]

    M = np.zeros((N, N))
    for i in range(num_pos_eigenvalues):
        v = selected_eigenvectors[:, i]
        M[:, 2*i] = np.real(v)
        M[:, 2*i+1] = np.imag(v)

    if num_pos_eigenvalues * 2 < N:
        zero_indices = np.where(np.abs(eigenvalues) < 1e-10)[0]
        null_space_vectors = np.real(eigenvectors[:, zero_indices])
        M[:, 2*num_pos_eigenvalues:] = null_space_vectors[:, :N - 2*num_pos_eigenvalues]

    M_inv = np.linalg.inv(M)
    C_canonical_unnormalized = M_inv @ C_real @ M
    
    J = np.eye(N)
    for i in range(num_pos_eigenvalues):
        lambda_val = C_canonical_unnormalized[2*i, 2*i+1]
        if abs(lambda_val) > 1e-10:
            norm_factor = 1.0 / np.sqrt(abs(lambda_val))
            J[2*i, 2*i] = norm_factor
            J[2*i+1, 2*i+1] = norm_factor
            
    final_C = J @ C_canonical_unnormalized @ J
    final_C[np.abs(final_C) < 1e-6] = 0

    return final_C, J, M

# ==============================================================================
#  DEINE FINALE FUNKTION
# ==============================================================================
def transform_complex_C(C_complex):
    # ... (keine Änderung)
    if not np.allclose(np.real(C_complex), 0):
        raise ValueError("Die Eingabematrix C muss rein imaginär sein.")

    C_real = np.real(C_complex / 1j)

    final_C, J, M = _transform_real_skew_matrix(C_real)
    
    return final_C, J, M

# ==============================================================================
#  HAUPTSKRIPT: ANWENDUNG
# ==============================================================================
# if __name__ == "__main__":
#     lambda_values = [1, 2, 0, 5, 0, 0, 8, 4]
#     C_num_sympy = get_C_matrix(lambda_values)
#     C_complex = np.array(C_num_sympy.tolist(), dtype=complex)

#     # Rufe die finale Funktion auf
#     final_C_real, J, M = transform_complex_C(C_complex)

#     # *** HIER IST DIE ANPASSUNG ***
#     # Wandle das reelle Ergebnis zurück in die komplexe kanonische Form
#     final_C_complex = final_C_real * 1j

#     # Ergebnisse ausgeben
#     print("✅ Finale komplexe transformierte Matrix:")
#     pprint(Matrix(final_C_complex))
#     print("-" * 50)
