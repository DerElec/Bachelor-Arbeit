"""
Build the 10×10 matrix Q for SU(3)  (revised, June-2025)
– Final result shown as (Gamma/4) · M  with symbolic Gamma
– LaTeX output keeps square roots in symbolic form.

Author  : ChatGPT
Requires: numpy ≥ 1.24, sympy
"""

import numpy as np
import sympy as sp
from typing import Tuple

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
PHYS_DIM = 8     # physical SU(3) generator space
DIM      = 10    # output matrix dimension (pad with empty entries)

# ----------------------------------------------------------------------
# 1.  Gell-Mann matrices λ¹ … λ⁸ (numerical; √3 will be recovered later)
# ----------------------------------------------------------------------
λ1 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 0]], dtype=complex)
λ2 = np.array([[0, -1j, 0],
               [1j,  0, 0],
               [0,   0, 0]], dtype=complex)
λ3 = np.array([[1,  0, 0],
               [0, -1, 0],
               [0,  0, 0]], dtype=complex)
λ4 = np.array([[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]], dtype=complex)
λ5 = np.array([[0,   0, -1j],
               [0,   0,   0],
               [1j,  0,   0]], dtype=complex)
λ6 = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0]], dtype=complex)
λ7 = np.array([[0,   0,   0],
               [0,   0, -1j],
               [0, 1j,   0]], dtype=complex)
λ8 = np.array([[ 1/np.sqrt(3), 0,               0],
               [ 0,             1/np.sqrt(3),   0],
               [ 0,             0,            -2/np.sqrt(3)]],
              dtype=complex)

GELL_MANN = [λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8]

# ----------------------------------------------------------------------
# 2.  Structure constants f^{abc}, d^{abc}
# ----------------------------------------------------------------------

def build_structure_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Return (f, d) with f[a,b,c] = f^{abc} and d[a,b,c] = d^{abc}."""
    f = np.zeros((PHYS_DIM, PHYS_DIM, PHYS_DIM))
    d = np.zeros_like(f)

    for a in range(PHYS_DIM):
        La = GELL_MANN[a]
        for b in range(PHYS_DIM):
            Lb = GELL_MANN[b]
            comm    = La @ Lb - Lb @ La         # [λᵃ, λᵇ]
            anticom = La @ Lb + Lb @ La         # {λᵃ, λᵇ}
            for c in range(PHYS_DIM):
                Lc = GELL_MANN[c]
                f_val = (1/(4j)) * np.trace(comm    @ Lc)
                d_val = (1/4)   * np.trace(anticom @ Lc)
                f[a, b, c] = float(np.real_if_close(f_val, tol=1e-9))
                d[a, b, c] = float(np.real_if_close(d_val, tol=1e-9))
    return f, d

# ----------------------------------------------------------------------
# 3.  Raw hermitian matrix  Q  (Γ-factor will be applied later)
# ----------------------------------------------------------------------

def build_Q_raw() -> np.ndarray:
    """Return hermitian 10×10 matrix Q_{kα} **without** the Γ/4 factor.

    Only the upper-left 8×8 physical block is populated; the remaining rows
    and columns (indices 8,9) are kept zero to represent the ‘empty’ entries
    requested by the user.
    """
    f, d = build_structure_arrays()

    # Start with a full zero-filled 10×10 matrix
    Q = np.zeros((DIM, DIM), dtype=complex)  # Q[k, α]

    for alpha in range(PHYS_DIM):            # α = 0 … 7 (physical indices)
        for k in range(PHYS_DIM):            # k = 0 … 7 (physical indices)
            total = 0j
            for c in range(PHYS_DIM):
                # Short aliases ------------------------------------------------
                f_a1c = f[alpha, 0, c]; f_a2c = f[alpha, 1, c]
                f_1ac = f[0, alpha, c]; f_2ac = f[1, alpha, c]

                d_1ck = d[0, c, k]; f_1ck = f[0, c, k]
                d_2ck = d[1, c, k]; f_2ck = f[1, c, k]

                d_c1k = d[c, 0, k]; f_c1k = f[c, 0, k]
                d_c2k = d[c, 1, k]; f_c2k = f[c, 1, k]

                # Eight terms (revised signs) ----------------------------------
                term1 = 1j * f_a1c * (d_1ck + 1j * f_1ck)
                term2 =      f_a1c * (d_2ck + 1j * f_2ck)
                term3 = -    f_a2c * (d_1ck + 1j * f_1ck)
                term4 = 1j * f_a2c * (d_2ck + 1j * f_2ck)
                term5 = 1j * f_1ac * (d_c1k + 1j * f_c1k)
                term6 =    - f_1ac * (d_c2k + 1j * f_c2k)
                term7 =      f_2ac * (d_c1k + 1j * f_c1k)
                term8 = 1j * f_2ac * (d_c2k + 1j * f_c2k)
                total += term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

            Q[k, alpha] = total  # only the physical block is filled

    return Q  # still without Γ/4

# ----------------------------------------------------------------------
# 4.  Matrix post-processing and output
# ----------------------------------------------------------------------

def pretty_ascii(Q_clean: np.ndarray) -> None:
    """Print Q_phys in a human-friendly ASCII table (German headings)."""
    header = (
        "        α = 1            2            3            4            5            6            7            8            9           10"
    )
    print("\nHermitesche Matrix  Q_phys  =  (Γ / 4) · M\n")
    print(header)
    print("        " + "-" * (len(header) - 8))
    for k in range(DIM):
        row_elems = []
        for a in range(DIM):
            val = Q_clean[k, a]
            row_elems.append("     0     " if abs(val) < 1e-12 else f"{val:+10.4g}")
        print(f"k = {k+1:>2} | " + " ".join(row_elems))
    print("\n⇒  Ganze Matrix   Q_phys  erhält man durch Multiplikation der obigen Zahlen mit  Γ / 4 .")


def latex_matrix(Q_sym_exact: sp.Matrix) -> None:
    """Print a LaTeX representation of the matrix (German heading)."""
    print("\nLaTeX-Darstellung der Matrix  Q_phys  (inklusive Γ / 4):\n")
    print("\\[")
    print(sp.latex(Q_sym_exact, mat_str="bmatrix"))
    print("\\]\n")


# ----------------------------------------------------------------------
# 5.  Main block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 5.1   Build matrix without Γ/4
    Q0 = build_Q_raw()

    # 5.2   Replace tiny numbers by 0.0  (for the ASCII view)
    Q_clean = np.where(np.abs(Q0) < 1e-12, 0.0, Q0)

    # 5.3   Convert numeric entries to exact SymPy expressions
    allowed_consts = [sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)]
    Q_exact = sp.Matrix([[0] * DIM for _ in range(DIM)])
    for i in range(DIM):
        for j in range(DIM):
            val = Q0[i, j]
            Q_exact[i, j] = 0 if abs(val) < 1e-12 else sp.nsimplify(val, allowed_consts)

    # 5.4   Multiply by symbolic Γ / 4
    Γ = sp.Symbol('Gamma', real=True)
    Q_sym_exact = (Γ / 4) * Q_exact

    # 5.5   Outputs -----------------------------------------------------
    # Uncomment the desired output helpers as needed.
    # pretty_ascii(Q_clean)       # ASCII table
    # latex_matrix(Q_sym_exact)   # LaTeX with symbolic roots

    # Always print the raw SymPy matrix (10×10) for further processing
    print(Q_sym_exact)