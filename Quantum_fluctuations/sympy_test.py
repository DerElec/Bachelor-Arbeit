# English comments; German console text
import sympy as sp
sp.init_printing(use_unicode=True, wrap_line=False, num_columns=1000)

def pprint_ascii(expr):
    # Force ASCII-only, no wrapping
    s = sp.pretty(expr, use_unicode=False, wrap_line=False, num_columns=10**6)
    print(s)

# Example
G, g0, m10, D1 = sp.symbols('G g0 m10 D1')
M = sp.Matrix([[G/2, sp.sqrt(2)*g0*m10/2],
               [-D1, -G/2]])
pprint_ascii(M)  # purely ASCII, cannot mojibake

# English comments; German console text
import sys, os, shutil

print("isatty:", sys.stdout.isatty())
print("stdout encoding:", sys.stdout.encoding)
print("TERM_PROGRAM:", os.environ.get("TERM_PROGRAM"))
print("VSCODE_PID:", os.environ.get("VSCODE_PID"))
try:
    cols = shutil.get_terminal_size().columns
except Exception:
    cols = None
print("terminal columns:", cols)

print("\nUnicode sanity:")
print("Ω √2 ⋅ κ η ┌─┐")  # if this breaks, viewer decodes wrong

print("\nBytes of 'Ω':", "Ω".encode("utf-8"))
