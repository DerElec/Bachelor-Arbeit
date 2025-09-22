import numpy as np

# ===============================
# Utilities: numerics & checking
# ===============================

def is_hermitian(M, tol=1e-10):
    """Check Hermiticity of a complex matrix."""
    return np.allclose(M, M.conj().T, atol=tol)

def min_eigval_hermitian(M):
    """Return minimal eigenvalue of a Hermitian matrix."""
    vals = np.linalg.eigvalsh(M)
    return float(np.min(np.real_if_close(vals)))

# ===========================================
# SU(3) / generators: f,d tensors and singles
# ===========================================

def su3_structure_constants_from_generators(lams):
    """
    Compute f_{abc} and d_{abc} from given 3x3 generators 'lams' (list of length 8).
    This works for standard Gell-Mann λ_a *or* any rescaled generators T_a,
    because we use trace identities directly:
        f_{abc} = (1/4i) Tr([G_a, G_b] G_c)
        d_{abc} = (1/4)  Tr({G_a, G_b} G_c)
    The normalization (e.g., Tr λ_a λ_b = 2 δ_ab) is implicitly respected.
    """
    f = np.zeros((8, 8, 8), dtype=float)
    d = np.zeros((8, 8, 8), dtype=float)
    for a in range(8):
        for b in range(8):
            comm = lams[a] @ lams[b] - lams[b] @ lams[a]
            anti = lams[a] @ lams[b] + lams[b] @ lams[a]
            for c in range(8):
                # Real-part projection suppresses ~1e-16 imag noise
                f[a, b, c] = np.real(np.trace(comm @ lams[c]) / (4j))
                d[a, b, c] = np.real(np.trace(anti @ lams[c]) / 4.0)
    return f, d

def means_from_density(lams, rho):
    """
    Compute singles m_a = Tr(rho * G_a) for the 8 generators provided.
    Works for pure or mixed rho (3x3 density matrix).
    """
    m = np.zeros(8, dtype=float)
    for a in range(8):
        m[a] = float(np.real(np.trace(rho @ lams[a])))
    return m

# ========================================
# Build SU(3) covariance and symplectic s
# ========================================

def sigma_su3_block(m, d_tensor, norm_const=2.0/3.0):
    """
    Build SU(3) covariance block for the 8 generators.
    Formula (for standard λ): Σ_ab = 2/3 δ_ab + d_{abc} m_c - m_a m_b
    If you pass scaled generators, the (2/3) factor effectively adapts via the d-tensor.
    Keep norm_const=2/3 for standard λ. For T=λ/2 you would use 1/3.
    """
    m = np.asarray(m, dtype=float)
    I = norm_const * np.eye(8)
    dmc = np.tensordot(d_tensor, m, axes=(2, 0))  # shape (8,8)
    return np.real_if_close(I + dmc - np.outer(m, m))

def s_su3_block(m, f_tensor, commutator_coeff=2.0):
    """
    s_ab = -i<[G_a, G_b]>.
    If [λ_a, λ_b] = 2i f_{abc} λ_c, then s_ab = 2 f_{abc} m_c  (commutator_coeff=2).
    For generators T_a = λ_a/2, [T_a,T_b] = i f_{abc} T_c -> use commutator_coeff=1.
    """
    m = np.asarray(m, dtype=float)
    return np.real_if_close(commutator_coeff * np.tensordot(f_tensor, m, axes=(2, 0)))

# ===================================
# Cavity (Q,P) covariance and s block
# ===================================

def sigma_qp_block(q_mean=0.0, p_mean=0.0, var_q=0.5, var_p=0.5, cov_qp=0.0):
    """
    Generic (Q,P) covariance block:
      Σ_QP = [[Var(Q), Cov(Q,P)],
              [Cov(Q,P), Var(P)]]
    Means (q_mean, p_mean) are not needed for Σ itself, but you might want them returned.
    """
    Sig = np.array([[var_q,  cov_qp],
                    [cov_qp, var_p]], dtype=float)
    return Sig

def s_qp_block(qp_scale=1.0):
    """
    Symplectic block for (Q,P) when [Q,P] = i * qp_scale.
    Standard canonical scaling -> qp_scale = 1.
    """
    return np.array([[0.0,  qp_scale],
                     [-qp_scale, 0.0]], dtype=float)

# ===========================
# Full assembly and top-level
# ===========================

def assemble_sigma_and_s(lams, rho_atom,
                         q_mean=0.0, p_mean=0.0,
                         var_q=0.5, var_p=0.5, cov_qp=0.0,
                         qp_scale=1.0,
                         norm_const_lambda=2.0/3.0,
                         commutator_coeff_lambda=2.0):
    """
    Assemble Σ and s for X = (G_1..G_8, Q, P), with product initial state ρ_atom ⊗ ρ_cav.
    Args:
      lams: list of length-8 3x3 numpy arrays (your generators, e.g., standard λ).
      rho_atom: 3x3 density matrix (Hermitian, PSD, Tr=1).
      q_mean, p_mean: cavity means (do not affect Σ, but returned for completeness).
      var_q, var_p, cov_qp: cavity covariance entries.
      qp_scale: [Q,P] = i * qp_scale  (1.0 for canonical).
      norm_const_lambda: 2/3 for λ; 1/3 for T=λ/2.
      commutator_coeff_lambda: 2 for λ; 1 for T=λ/2.
    Returns:
      Sigma (10x10), s (10x10), m (length-10 means vector in order (G1..G8,Q,P))
    """
    # Compute f,d from the provided generators (safe against normalization mismatches)
    f, d = su3_structure_constants_from_generators(lams)

    # Singles for SU(3) part
    m8 = means_from_density(lams, rho_atom)

    # Build SU(3) blocks
    Sig_l = sigma_su3_block(m8, d, norm_const=norm_const_lambda)
    s_l   = s_su3_block(m8, f, commutator_coeff=commutator_coeff_lambda)

    # Build cavity blocks
    Sig_qp = sigma_qp_block(q_mean=q_mean, p_mean=p_mean,
                            var_q=var_q, var_p=var_p, cov_qp=cov_qp)
    s_qp   = s_qp_block(qp_scale=qp_scale)

    # Assemble full matrices
    Sigma = np.zeros((10, 10), dtype=float)
    s_mat = np.zeros((10, 10), dtype=float)
    Sigma[:8, :8] = Sig_l
    Sigma[8:, 8:] = Sig_qp
    s_mat[:8, :8] = s_l
    s_mat[8:, 8:] = s_qp

    # Means vector (G1..G8,Q,P)
    m_full = np.zeros(10, dtype=float)
    m_full[:8] = m8
    m_full[8]  = q_mean
    m_full[9]  = p_mean

    return Sigma, s_mat, m_full

def psd_check_report(Sigma, s_mat, eps=1e-10):
    """
    Check Hermiticity and PSD of M = Sigma + i/2 s, and print a German report.
    """
    M = Sigma + 0.5j * s_mat
    print("=== Konsistenz-Check: Σ(0) + i/2 · s(0) ===")
    print(f"Hermitizität: {is_hermitian(M)}")
    lam_min = min_eigval_hermitian(M)
    print(f"Kleinster Eigenwert: {lam_min:.12e}")
    print(f"PSD (≥ -{eps:g}): {lam_min >= -eps}")
    return lam_min

# =========================================
# Example usage (customize to your project)
# =========================================

if __name__ == "__main__":
    # --- Example generators: standard Gell-Mann λ (Tr λ_a λ_b = 2 δ_ab) ---
    def gell_mann_standard():
        l = []
        l.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))                     # λ1
        l.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex))                   # λ2
        l.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))                     # λ3
        l.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))                      # λ4
        l.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))                   # λ5
        l.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))                      # λ6
        l.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))                   # λ7
        l.append((1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex))      # λ8
        return l

    lams = gell_mann_standard()

    # --- Example atomic state: populations ρ_00=0.8, ρ_11=0.0, ρ_22=0.2 (diagonal) ---
    rho = np.diag([1, 0.0, 0.0]).astype(complex)

    # --- Example cavity: non-vacuum means and generic variances ---
    q_mean, p_mean = 0.15, -0.10     # e.g., displaced coherent-like mean
    var_q, var_p = 0.5, 0.5          # vacuum/coherent variances
    cov_qp = 0.0
    qp_scale = 1.0                   # [Q,P] = i * 1

    # --- Assemble Σ and s (for standard λ: norm_const=2/3, commutator_coeff=2) ---
    Sigma, s_mat, m_vec = assemble_sigma_and_s(
        lams, rho,
        q_mean=q_mean, p_mean=p_mean,
        var_q=var_q, var_p=var_p, cov_qp=cov_qp,
        qp_scale=qp_scale,
        norm_const_lambda=2.0,
        commutator_coeff_lambda=2.0
    )

    # --- Report & checks ---
    print("Singles m (G1..G8, Q, P):")
    print(np.array2string(m_vec, precision=6, suppress_small=True))
    _ = psd_check_report(Sigma, s_mat, eps=1e-10)

    # Optional: show small blocks for quick inspection
    print("\nSU(3)-Block von Σ (erste 4x4):")
    print(np.array2string(Sigma[:4,:4], precision=5, suppress_small=True))
    print("\n(Q,P)-Block von Σ:")
    print(np.array2string(Sigma[8:,8:], precision=5, suppress_small=True))
    print("\n(Q,P)-Block von s:")
    print(np.array2string(s_mat[8:,8:], precision=5, suppress_small=True))
