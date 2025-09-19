# Python code (comments in English; console/output in German)
if __name__ == "__main__":
    import numpy as np
    from multi_omega_v_unified_fixed import run_grid, SolveConfig, PhysParams

    omega_values = np.linspace(0.0, 12.0, 120)      # [4.0, 5.0, 6.0]
    v_values     = np.linspace(8.0, -8.0, 160)    # [-2.0, -2.4, ..., -4.0]

    cfg = SolveConfig(
        params=PhysParams(
            g0=1.0, Delta1=1.0, Delta2=1.0,
            gamma=1.0, kappa=1.0, eta=1.0,
            Omega=omega_values[0], V=v_values[0],   # overridden per run
            Gamma=2.0
        ),
        t_min=0.0, t_max=5000.0, num_points=2001,
        rtol=1e-8, atol=1e-8, method="RK45"
    )

    outfile = "heatmap_data_grid.h5"
    print("Starte Grid-Lauf …")
    run_grid(omega_values, v_values, outfile, cfg=cfg, max_workers=None, use_threads=False)
    print(f"Fertig. Datei geschrieben: {outfile}")
