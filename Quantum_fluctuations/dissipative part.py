import sympy as sp

# Define symbolic variables for gamma and m_i (i=1,...,8)
gamma = sp.symbols('gamma')
m = sp.symbols('m1:9')  # creates m1, m2, ..., m8

# Define the imaginary unit as I from sympy
I = sp.I

# Define the 8 Gell-Mann matrices for SU(3) as 3x3 sympy Matrix objects.
lambda1 = sp.Matrix([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])

lambda2 = sp.Matrix([[0, -I, 0],
                       [I,  0, 0],
                       [0,  0, 0]])

lambda3 = sp.Matrix([[1,  0, 0],
                       [0, -1, 0],
                       [0,  0, 0]])

lambda4 = sp.Matrix([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 0]])

lambda5 = sp.Matrix([[0, 0, -I],
                       [0, 0,  0],
                       [I, 0,  0]])

lambda6 = sp.Matrix([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]])

lambda7 = sp.Matrix([[0, 0, 0],
                       [0, 0, -I],
                       [0, I, 0]])

lambda8 = sp.Matrix([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -2]]) * (1/sp.sqrt(3))

# Create a list of Gell-Mann matrices (λ_a)
lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]

# Compute the generators T_a = lambda_a / 2
T = [mat / 2 for mat in lambdas]

# Function to compute the structure constant f^{abc} using the formula:
# f^{abc} = -2*i*Tr([T_a, T_b]*T_c)
def compute_structure_constant(a, b, c):
    """
    Compute the structure constant f^{abc} for indices a, b, c (1-indexed).
    """
    Ta = T[a - 1]
    Tb = T[b - 1]
    Tc = T[c - 1]
    commutator = Ta * Tb - Tb * Ta
    trace_val = sp.trace(commutator * Tc)
    f_val = -2 * I * trace_val
    return sp.simplify(f_val)

# Precompute all structure constants f^{abc} for a,b,c = 1,...,8 and store in a dictionary
f_constants = {}
for a in range(1, 9):
    for b in range(1, 9):
        for c in range(1, 9):
            f_constants[(a, b, c)] = compute_structure_constant(a, b, c)

# Function to retrieve f^{abc} from the dictionary (1-indexed)
def f(a, b, c):
    """
    Retrieve the computed structure constant f^{abc} for indices a, b, c.
    """
    return f_constants.get((a, b, c), 0)

# Compute the symbolic s_ab matrix using the formula:
# s_{ab} = sum_{c,d} gamma * m_c * m_d *
#         ( f^{1ac}*f^{b1d} + I*f^{1ac}*f^{b2d} - I*f^{2ac}*f^{b1d} + f^{2ac}*f^{b2d} )
def compute_Z_prime_matrix():
    # Create an 10x10 symbolic matrix (a, b from 1 to 10)
    s = sp.Matrix(10, 10, lambda i, j: 0)
    for a in range(1, 9):         # a: row index
        for b in range(1, 9):     # b: column index
            expr = 0
            for c in range(1, 9):  # summing index c
                for d in range(1, 9):  # summing index d
                    term1 = f(1, a, c) * f(b, 1, d)         # f^{1ac} * f^{b1d}
                    term2 = I * f(1, a, c) * f(b, 2, d)       # i * f^{1ac} * f^{b2d}
                    term3 = -I * f(2, a, c) * f(b, 1, d)      # -i * f^{2ac} * f^{b1d}
                    term4 = f(2, a, c) * f(b, 2, d)           # f^{2ac} * f^{b2d}
                    expr += gamma * m[c - 1] * m[d - 1] * (term1 + term2 + term3 + term4)
            s[a - 1, b - 1] = sp.simplify(expr)
    return s

# Compute the symbolic s_ab matrix
Z_prime_matrix = compute_Z_prime_matrix()

# Calculate the symmetrized matrix Z_ab = (s + s^T)/2
z_matrix = (Z_prime_matrix + Z_prime_matrix.T) / 2

# Further simplify each entry of z_matrix for the simplest form
z_matrix_simple = sp.Matrix(z_matrix.shape[0], z_matrix.shape[1],
                            lambda i, j: sp.simplify(z_matrix[i, j]))

# Output the symbolic matrix Z_ab and its LaTeX representation
print("Die symbolische Matrix Z_ab lautet:")
sp.pprint(z_matrix_simple)
print("\nLaTeX Code der Matrix Z_ab:")
latex_code_Z = sp.latex(z_matrix_simple)
print(latex_code_Z)
