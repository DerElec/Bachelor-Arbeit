import sympy as sp
# Force UTF-8 output for the process
import sys, io, os
if hasattr(sys.stdout, "reconfigure"):          # Python 3.7+
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:                                           # Fallback for very old Pythons
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

os.environ["PYTHONIOENCODING"] = "utf-8"        # Belt-and-suspenders

    # Definiere dein Symbol
Omega = sp.Symbol('Ω')

# Initialisiere das "pretty printing" mit Unicode-Unterstützung
sp.init_printing(use_unicode=True)

# Jetzt wird das Symbol korrekt ausgegeben
sp.pprint(Omega)        # zeigt schön Ω
sp.pprint(2 * Omega)    # zeigt 2⋅Ω