import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ------------ Daten laden -----------------------------------------
df = pd.read_pickle("results_delta_1_060825_multi.pkl")

# ------------ Ω extrahieren ---------------------------------------
df["Omega"] = df["additional params"].apply(lambda x: x[3])

# ------------ Varianz-Summe bilden --------------------------------
var_cols = [c for c in df.columns if c.startswith("var_")]
df["sum_var"] = df[var_cols].sum(axis=1)

# ------------ Pivot-Tabellen --------------------------------------
pivot_m11 = (
    df.pivot_table(index="V", columns="Omega", values="<1|1>", aggfunc="mean")
      .sort_index()
      .sort_index(axis=1)
)

pivot_var = (
    df.pivot_table(index="V", columns="Omega", values="sum_var", aggfunc="mean")
      .sort_index()
      .sort_index(axis=1)
)

# ------------ Komplexe → Reelle Daten -----------------------------
def to_real(arr, name):
    """Convert array to float, warn if imag part > 1e-8."""
    imag_max = np.max(np.abs(np.imag(arr[np.isfinite(arr)])))  # ignore NaN
    if imag_max > 1e-8:
        warnings.warn(
            f"{name}: max(|Im|) = {imag_max:.2e} ist nicht vernachlässigbar.",
            RuntimeWarning,
        )
    return np.real(arr)

data_m11 = to_real(pivot_m11.values, "rho11_mean")
data_var = to_real(pivot_var.values, "sum_var")

# ------------ Phase-Diagramme plotten -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# -- 1) Mittelwert ρ11 --------------------------------------------
im0 = axes[0].imshow(
    data_m11,
    origin="lower",
    aspect="auto",
    extent=[
        pivot_m11.columns.min(),
        pivot_m11.columns.max(),
        pivot_m11.index.min(),
        pivot_m11.index.max(),
    ],
)
axes[0].set_title(r"Mittelwert von $\rho_{11}$")
axes[0].set_xlabel(r"$\Omega$")
axes[0].set_ylabel(r"$V$")
fig.colorbar(im0, ax=axes[0], label=r"$\langle 1|1 \rangle$")

# -- 2) Summe der Varianzen ---------------------------------------
im1 = axes[1].imshow(
    data_var,
    origin="lower",
    aspect="auto",
    extent=[
        pivot_var.columns.min(),
        pivot_var.columns.max(),
        pivot_var.index.min(),
        pivot_var.index.max(),
    ],
)
axes[1].set_title("Summe aller Varianzen")
axes[1].set_xlabel(r"$\Omega$")
axes[1].set_ylabel(r"$V$")
fig.colorbar(im1, ax=axes[1], label=r"$\sum \mathrm{Var}$")

plt.show()
