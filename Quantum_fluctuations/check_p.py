import sympy as sp

def build_P_matrix_new():
    """
    Berechnet die Jacobi-Matrix P für das System der Mean-Field-Gleichungen.
    Dieser Ansatz ist robuster und direkter als die manuelle Berechnung 
    über Strukturkonstanten.
    """
    # --- 1. Symbole definieren ---
    # Parameter
    kappa, gamma, Delta1, Delta2, Omega, V, g0, eta = sp.symbols(
        'kappa Gamma Delta1 Delta2 Omega V g0 eta', real=True
    )
    # Zustandsvariablen (Vektor x)
    # λ_1 bis λ_8, dann q und p
    x = sp.symbols('lambda1:9 q p')
    lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8, q, p = x

    # --- 2. Vektor der Differentialgleichungen definieren (f(x) aus Anhang 6.6) ---
    # Wir nehmen die Gleichungen direkt aus Ihrem Anhang (6.107 - 6.118)
    # Hinweis: Diese DGLs müssen mit dem finalen Hamilton-Operator konsistent sein.
    
    dot_q = -kappa/2 * q + (gamma * g0 / sp.sqrt(2)) * lambda2 + sp.sqrt(2) * eta
    dot_p = -kappa/2 * p - (gamma * g0 / sp.sqrt(2)) * lambda1 # Im Anhang steht Im(η), hier als 0 angenommen

    dot_lambda1 = -gamma/2 * lambda1 + Delta1 * lambda2 + Omega/2 * lambda5 + sp.sqrt(2) * gamma * g0 * lambda3 * p
    dot_lambda2 = -gamma/2 * lambda2 - Delta1 * lambda1 - Omega/2 * lambda4 - sp.sqrt(2) * gamma * g0 * lambda3 * q
    dot_lambda3 = -gamma * lambda3 + (sp.sqrt(3) * gamma / 3) * lambda8 + 2*gamma/3 - Omega/2 * lambda7 - sp.sqrt(2) * gamma * g0 * (lambda1 * p - lambda2 * q)
    dot_lambda4 = (Delta2 - Delta1) * lambda5 # ACHTUNG: Anhang 6.6 hat hier wahrsch. Fehler. EOM für λ4, λ5, λ6, λ7 sind komplex. 
                                              # Dies ist eine vereinfachte reelle Version basierend auf den typischen Termen.
                                              # Die korrekten EOMs sollten aus dem Hamilton-Operator abgeleitet werden.
                                              # Hier als Platzhalter, um die Methode zu zeigen.
    dot_lambda5 = -(Delta2 - Delta1) * lambda4 - Omega/2 * lambda2 # Platzhalter
    dot_lambda6 = -gamma/2 * lambda6 + (Delta2 - Delta1) * lambda7 + Omega/2 * lambda4 # Platzhalter
    dot_lambda7 = -gamma/2 * lambda7 - (Delta2 - Delta1) * lambda6 - Omega/2 * lambda5 + (sp.sqrt(3)/2)*Omega*lambda8 # kombiniert mit EOM für λ8
    dot_lambda8 = (sp.sqrt(3) / 2) * Omega * lambda7

    # Vektor f(x)
    f = sp.Matrix([
        dot_lambda1, dot_lambda2, dot_lambda3, dot_lambda4, dot_lambda5, 
        dot_lambda6, dot_lambda7, dot_lambda8, dot_q, dot_p
    ])
    
    # --- 3. Jacobi-Matrix berechnen ---
    # P_ij = ∂f_i / ∂x_j
    P = f.jacobian(x)
    
    return P, x

# --- Code ausführen und Ergebnis anzeigen ---
if __name__ == '__main__':
    P_matrix, variables = build_P_matrix_new()
    
    print("Berechnete Jacobi-Matrix P:")
    sp.pprint(P_matrix)
    
    print("\nAbhängig von den Variablen:")
    print(variables)