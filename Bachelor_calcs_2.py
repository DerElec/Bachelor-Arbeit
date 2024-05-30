from sympy import symbols, I, Function, Eq, solve, conjugate
import numpy as np
# Define the variables and functions
kappa, gamma, Gamma, Omega, delta_2, delta_1, V = symbols('kappa gamma Gamma Omega delta_2 delta_1 V')
a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21 = symbols('a a_dagger psi00 psi11 psi22 psi01 psi10 psi02 psi20 psi12 psi21')


# a = Function('a')()  # Annihilation operator
# a_dagger = Function('a_dagger')()  # Creation operator
# psi00 = Function('psi00')()  # |0><0|
# psi11 = Function('psi11')()  # |1><1|
# psi22 = Function('psi22')()  # |2><2|
# psi01 = Function('psi01')()  # |0><1|
# psi10 = Function('psi10')()  # |1><0|
# psi02 = Function('psi02')()  # |0><2|
# psi20 = Function('psi20')()  # |2><0|
# psi12 = Function('psi12')()  # |1><2|
# psi21 = Function('psi21')()  # |2><1|

conjugate_relations = {
    conjugate(a): a_dagger,
    conjugate(a_dagger): a,
    conjugate(psi01): psi10,
    conjugate(psi10): psi01,
    conjugate(psi21): psi12,
    conjugate(psi12): psi21,
    conjugate(psi02): psi20,
    conjugate(psi20): psi02,
    # Selbstkonjugierte Zustände
    conjugate(psi00): psi00,
    conjugate(psi11): psi11,
    conjugate(psi22): psi22,
    #selbstkonj variablen
    conjugate(kappa): kappa,
    conjugate(gamma): gamma,
    conjugate(Gamma): Gamma,
    conjugate(Omega): Omega,
    conjugate(delta_2): delta_2,
    conjugate(delta_1): delta_1,
    conjugate(V): V
}
def apply_conjugate_relations(expression):
    """Ersetze alle konjugierten Symbole gemäß den definierten Beziehungen in einem Ausdruck."""
    return expression.subs(conjugate_relations)


# Stationary equations with the derivatives set to zero
eq1 = Eq(0, -kappa/2 * a - I * gamma * psi01)
eq1_conj = apply_conjugate_relations(Eq(conjugate(eq1.lhs), conjugate(eq1.rhs)))


eq2 = Eq(0, Gamma * psi11 + I * gamma * (psi10 * a - psi01 * a_dagger))
eq2_conj =apply_conjugate_relations(Eq(conjugate(eq2.lhs), conjugate(eq2.rhs)))

eq3 = Eq(0, -Gamma * psi11 + I * gamma * (psi01 * a_dagger - psi10 * a) + I * Omega/2 * (psi12 - psi21))
eq3_conj =apply_conjugate_relations(Eq(conjugate(eq3.lhs), conjugate(eq3.rhs)))

eq4 = Eq(0, I * Omega/2 * (psi12 - psi21))
eq4_conj =apply_conjugate_relations(Eq(conjugate(eq4.lhs), conjugate(eq4.rhs)))

eq5 = Eq(0, -Gamma/2 * psi21 + I * (delta_2 - delta_1) * psi21 - I * gamma * psi20 * a + I * Omega/2 * (psi11 - psi22) + 2 * V * psi21 * psi22)
eq5_conj =apply_conjugate_relations(Eq(conjugate(eq5.lhs), conjugate(eq5.rhs)))

eq6 = Eq(0, -Gamma/2 * psi01 + I * (-delta_1 * psi01 + gamma * (psi11 * a - psi00 * a_dagger) + Omega/2 * (-psi02)))
eq6_conj =apply_conjugate_relations(Eq(conjugate(eq6.lhs), conjugate(eq6.rhs)))

eq7 = Eq(0, I * (delta_2 * psi20 + Omega/2 * psi10 + 2 * V * psi20 * psi22 - gamma * psi21 * a_dagger))
eq7_conj =apply_conjugate_relations(Eq(conjugate(eq7.lhs), conjugate(eq7.rhs)))

print(eq1.rhs)
print(apply_conjugate_relations(conjugate(eq1.rhs)))


# Solve the system of equations (if solvable)
solutions = solve([eq1_conj, eq2_conj, eq3_conj, eq4_conj, eq5_conj, eq6_conj, eq7_conj,
                   eq1,eq2,eq3,eq4,eq5,eq6,eq7]
                  , (a, a_dagger, psi00, psi11, psi22, psi01, psi10, psi02, psi20, psi12, psi21), dict=True)
print(solutions)







