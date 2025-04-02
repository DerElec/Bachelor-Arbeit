import sympy as sp
from sympy import sqrt

# Define symbols for the parameters (all symbols are real)
hbar, Delta1, Omega, V, N, eta = sp.symbols('hbar Delta1 Omega V N eta', real=True)

# Define the full 10-component omega vector:
# omega = hbar*(0, 0, -Delta1, 0, 0, Omega, 0, -4V/(3sqrt(3)), 0, 2sqrt(N)*eta)
omega = {
    1: 0,
    2: 0,
    3: -hbar * Delta1,
    4: 0,
    5: 0,
    6: hbar * Omega,
    7: 0,
    8: -hbar * 4 * V / (3 * sp.sqrt(3)),
    9: 0,
    10: hbar * 2 * sp.sqrt(N) * eta
}

# Independent non-zero structure constants f^{abc} for the Gell-Mann matrices.
# Only the independent values are provided.
independent_f = {
    (1, 2, 3): 1,
    (1, 4, 7): sp.Rational(1, 2),
    (1, 5, 6): -sp.Rational(1, 2),
    (2, 4, 6): sp.Rational(1, 2),
    (2, 5, 7): sp.Rational(1, 2),
    (3, 4, 5): sp.Rational(1, 2),
    (3, 6, 7): -sp.Rational(1, 2),
    (4, 5, 8): sp.sqrt(3)/2,
    (6, 7, 8): sp.sqrt(3)/2
}

def permutation_sign(perm):
    """
    Compute the sign of a permutation given as a tuple.
    """
    perm_list = list(perm)
    sign = 1
    for i in range(len(perm_list)):
        for j in range(i+1, len(perm_list)):
            if perm_list[i] > perm_list[j]:
                sign *= -1
    return sign

def f(i, a, c):
    """
    Compute the structure constant f^{i a c} using the independent components and antisymmetry.
    """
    # Loop over all independent components and check if (i, a, c) is a permutation of one of them.
    for (p, q, r), value in independent_f.items():
        indices = (p, q, r)
        if sorted((i, a, c)) == sorted(indices):
            # Determine the sign for the permutation required to go from (p, q, r) to (i, a, c)
            base_list = list(indices)
            target_list = [i, a, c]
            perm = [base_list.index(elem) for elem in target_list]
            sign = permutation_sign(tuple(perm))
            return sign * value
    # If not found, the value is 0.
    return 0

# Create a 10x10 matrix M with indices (c, a).
# For c, a in {1,...,8} compute:
#   M(c,a) = -2 * sum_{i=1}^{8} [omega[i] * f(i, a, c)]
# For other indices (c or a in {9,10}) set the entry to 0.
M = sp.Matrix(10, 10, lambda row, col: 0)

for a in range(1, 11):    # a = 1,...,10 (columns)
    for c in range(1, 11):  # c = 1,...,10 (rows)
        if a <= 8 and c <= 8:
            # Sum over i = 1,...,8
            s = sum(omega[i] * f(i, a, c) for i in range(1, 9))
            M[c-1, a-1] = -2 * s
        else:
            M[c-1, a-1] = 0  # Set entries to 0 for indices > 8

# Initialize pretty printing and display the matrix.
sp.init_printing()
print("Die 10x10 Matrix M(c,a) lautet:")
sp.pprint(M)

# Generate LaTeX code of the matrix
latex_code = sp.latex(M)
print("\nDer LaTeX-Code der Matrix lautet:")
print(latex_code)
