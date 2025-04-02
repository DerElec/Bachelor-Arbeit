import sympy as sp

# Symbole definieren
p, g0, Delta1, Delta2, Omega, eta, V, hbar = sp.symbols('p g0 Delta1 Delta2 Omega eta V hbar', real=True)
sqrt2 = sp.sqrt(2)

# Single-atom Hamiltonian in der P-Basis (semiklassische Näherung: Ableitungsterm vernachlässigt)
H1 = sp.Matrix([
    [0, -g0/sqrt2 * p, 0],
    [g0/sqrt2 * p, Delta1, Omega/2],
    [0, Omega/2, Delta2]
])

# 3x3 Einheitsmatrix
I3 = sp.eye(3)

# Zweiatom-Hamiltonian: Summe der Einzelatom-Hamiltonoperatoren (Tensorprodukt) plus Interaktion
H_total = sqrt2 * eta * p * sp.eye(9) + sp.kronecker_product(H1, I3) + sp.kronecker_product(I3, H1)

# Projektion auf den Zustand |2><2| für ein Atom (Zustandsordnung: 0, 1, 2)
proj2 = sp.Matrix([[0,0,0],[0,0,0],[0,0,1]])
H_int = hbar * V * sp.kronecker_product(proj2, proj2)

# Gesamt-Hamiltonian
H_total = H_total + H_int

# Eigenwerte berechnen
eigenvals = H_total.eigenvals()
sp.pprint(eigenvals)
