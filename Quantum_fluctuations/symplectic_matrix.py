from sympy import symbols, I, Matrix, zeros, sqrt, simplify
from itertools import permutations
import numpy as np

def get_S_matrix_gellman(v=None):

    lambda_syms = symbols('lambda1:9')
    f = {(0,1,2): 1, (0,3,6): 1/2, (1,3,5): 1/2, (1,4,6): -1/2,
         (2,3,4): 1/2, (2,5,6): 1/2, (3,4,7): sqrt(3)/2, (5,6,7): -sqrt(3)/2}
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

def expand_to_10x10(mat):
    """
    Expand an 8x8 matrix to a 10x10 matrix by padding zeros on the right and bottom,
    preserving the original entries and setting the two new diagonal entries to 1.
    """
    # Create a 10×10 zero matrix with the same dtype as the input
    new_mat = np.zeros((10, 10), dtype=mat.dtype)
    # Copy the original 8×8 matrix into the top-left corner
    new_mat[:8, :8] = mat
    # Set the two new diagonal entries (positions [8,8] and [9,9]) to 1
    for i in range(8, 10):
        new_mat[i, i] = 1
    return new_mat
def expand_to_10x10_sym(mat):
    """
    Expand an 8x8 matrix to a 10x10 matrix by padding zeros on the right and bottom,
    preserving the original entries and setting the two new diagonal entries to 1.
    """
    # Create a 10×10 zero matrix with the same dtype as the input
    new_mat = np.zeros((10, 10), dtype=mat.dtype)
    # Copy the original 8×8 matrix into the top-left corner
    new_mat[:8, :8] = mat
    # Set the two new diagonal entries (positions [8,8] and [9,9]) to 1
    new_mat[8, 9] = 1
    new_mat[9, 8] = -1
    return new_mat





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

def transform_complex_S(S_complex):

    if not np.allclose(np.real(S_complex), 0):
        raise ValueError("Die Eingabematrix C muss rein imaginär sein.")

    C_real = np.real(S_complex / 1j)

    final_C, J, M = _transform_real_skew_matrix(C_real)
    
    return final_C, J, M
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
                 simplify(sum(2 * 1j * f_full.get((i, j, k), 0) * lambda_syms[k]
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
