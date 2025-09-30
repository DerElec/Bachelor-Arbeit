# English comments; German console text
import sympy as sp
import shutil

def pprint_now(expr, cols=None):
    """Pretty-print without wrapping, using a very large width by default."""
    if cols is None:
        try:
            cols = shutil.get_terminal_size().columns
        except Exception:
            cols = 300
    # Force no wrapping and huge column budget
    s = sp.pretty(expr, use_unicode=True, wrap_line=False, num_columns=max(cols, 1000))
    print(s)

# Example:
# sp.init_printing(use_unicode=True)  # optional
# pprint_now( your_big_matrix )
