################################################################################
# Import required packages
using OrdinaryDiffEq       # For ODE solvers
using Plots                # For plotting heatmaps
using HDF5                 # For saving data in h5 files
using Statistics           # For mean and variance computations
using Base.Threads         # For multithreading

################################################################################
# Define the ODE function (with English comments)
function rhs_dgl(y, params)
    # Unpack state vector
    a          = y[1]
    a_dagger   = y[2]
    ket00      = y[3]
    ket01      = y[4]
    ket10      = y[5]
    ket11      = y[6]
    ket22      = y[7]
    ket21      = y[8]
    ket12      = y[9]
    ket20      = y[10]
    ket02      = y[11]

    # Unpack parameters
    κ     = params.kappa     # e.g. 1.0
    γ     = params.gamma     # e.g. 1.0
    Γ     = params.Gamma     # e.g. 2.0
    Ω     = params.Omega     # overwritten in the parameter scan
    δ₁    = params.delta1    # e.g. 1.0
    δ₂    = params.delta2    # e.g. 1.0
    η     = params.eta       # e.g. 1.0
    V     = params.V         # overwritten in the parameter scan

    # Define the ODEs
    da_dt          = -κ/2 * a - 1im*(γ*ket01) + η
    da_dagger_dt   = conj(da_dt)

    dket00_dt      = Γ*ket11 + 1im*γ*(ket10*a - ket01*a_dagger)
    dket01_dt      = -Γ/2 * ket01 + 1im*(-δ₁*ket01 + γ*(ket11*a - ket00*a) - Ω/2*ket02)
    dket10_dt      = conj(dket01_dt)

    dket11_dt      = -Γ*ket11 + 1im*γ*(ket01*a_dagger - ket10*a) + 1im*(Ω/2)*(ket21 - ket12)
    dket22_dt      = 1im*(Ω/2)*(ket12 - ket21)

    dket21_dt      = -Γ/2 * ket21 + 1im*(δ₂*ket21 - δ₁*ket21 - γ*ket20*a + (Ω/2)*(ket11 - ket22) + 2*V*ket21*ket22)
    dket12_dt      = conj(dket21_dt)

    dket02_dt      = 1im*(-δ₂*ket02 - Ω/2*ket01 - 2*V*ket02*ket22 + γ*ket12*a)
    dket20_dt      = conj(dket02_dt)

    return [
        da_dt,
        da_dagger_dt,
        dket00_dt,
        dket01_dt,
        dket10_dt,
        dket11_dt,
        dket22_dt,
        dket21_dt,
        dket12_dt,
        dket20_dt,
        dket02_dt
    ]
end

################################################################################
# Define fixed parameters (except Omega and V, which will be overwritten)
base_params = (
    kappa  = 1.0,
    gamma  = 1.0,    # Hier wird gamma auch als g₀ angenommen!
    Gamma  = 2.0,
    Omega  = 8.0,    # initial value (will be overwritten)
    delta1 = 1.0,
    delta2 = 1.0,
    eta    = 1.0,
    V      = -8.0    # initial value (will be overwritten)
)

################################################################################
# Define time span for the simulation
t0 = 0.0
t_end = 2000.0
tspan = (t0, t_end)

################################################################################
# Define simulation initial conditions (state vector with 11 components)
simulations = [
    # Simulation 1: Alle Komponenten 0 außer ψ₀₀ = 1 (Index 3)
    ("Sim1_psi00", [0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
    # Simulation 2: Alle Komponenten 0 außer ψ₁₁ = 1 (Index 6)
    ("Sim2_psi11", [0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im])
]

################################################################################
# Define parameter ranges for Omega and V.
# (Hinweis: Falls die analytische Linie negative V-Werte liefert, solltest Du 
# den V-Bereich anpassen. Hier bleiben die Bereiche, wie zuvor, als Beispiel.)
omega_values = collect(range(0.0, stop=8.0, length=20))  # x-Achse
v_values     = collect(range(0.0, stop=8.0, length=20))  # y-Achse

# Anzahl der Werte
n_omega = length(omega_values)
n_v     = length(v_values)

# Berechne die analytische Linie als Funktion von Omega (g_0 wird als gamma interpretiert)
line_V = -base_params.delta2/2 * (((omega_values .* base_params.kappa) ./ (4 * base_params.eta * base_params.gamma)).^2 .+ 1)

################################################################################
# Loop über die verschiedenen Initialbedingungen
for (sim_label, y0) in simulations
    println("starte Simulation: ", sim_label)
    
    # Pre-allocation der Arrays für Mittelwert und Varianz der 5 Observablen:
    # 1. |a| (Kavitätenfeld)
    mean_a_array = zeros(n_v, n_omega)
    var_a_array  = zeros(n_v, n_omega)
    
    # 2. Kavitätenpopulation |a|²
    mean_pop_array = zeros(n_v, n_omega)
    var_pop_array  = zeros(n_v, n_omega)
    
    # 3. ψ₀₀ (Index 3)
    mean_psi00_array = zeros(n_v, n_omega)
    var_psi00_array  = zeros(n_v, n_omega)
    
    # 4. ψ₁₁ (Index 6)
    mean_psi11_array = zeros(n_v, n_omega)
    var_psi11_array  = zeros(n_v, n_omega)
    
    # 5. ψ₂₂ (Index 7)
    mean_psi22_array = zeros(n_v, n_omega)
    var_psi22_array  = zeros(n_v, n_omega)
    
    # Parallelisierter Loop über die Parameter (V und Omega)
    @threads for i in 1:n_v
        for j in 1:n_omega
            v_val = v_values[i]
            omega = omega_values[j]
            # Update parameters with current Omega and V values
            current_params = (; base_params..., Omega = omega, V = v_val)
            prob = ODEProblem((y, p, t) -> rhs_dgl(y, p), y0, tspan, current_params)
            sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8)
            
            # Verwende die letzten 25 % der Zeitreihe für die Auswertung:
            n_t = length(sol.t)
            start_index = floor(Int, 0.75 * n_t) + 1
            
            # 1. a (Kavitätenfeld): Es wird der Betrag von a ausgewertet.
            last_quarter_a = sol[1, start_index:end]
            # 2. Kavitätenpopulation: |a|²
            last_quarter_pop = abs.(sol[1, start_index:end]).^2
            
            # 3. ψ₀₀ (Index 3), 4. ψ₁₁ (Index 6), 5. ψ₂₂ (Index 7) – als reelle Größen:
            last_quarter_psi00 = real.(sol[3, start_index:end])
            last_quarter_psi11 = real.(sol[6, start_index:end])
            last_quarter_psi22 = real.(sol[7, start_index:end])
            
            # Berechne Mittelwerte und Varianzen
            mean_a    = abs(mean(last_quarter_a))
            var_a     = var(abs.(last_quarter_a))
            
            mean_pop  = mean(last_quarter_pop)
            var_pop   = var(last_quarter_pop)
            
            mean_psi00 = mean(last_quarter_psi00)
            var_psi00  = var(last_quarter_psi00)
            
            mean_psi11 = mean(last_quarter_psi11)
            var_psi11  = var(last_quarter_psi11)
            
            mean_psi22 = mean(last_quarter_psi22)
            var_psi22  = var(last_quarter_psi22)
            
            # Speichere die Ergebnisse in den Arrays
            mean_a_array[i, j]    = mean_a
            var_a_array[i, j]     = var_a
            
            mean_pop_array[i, j]  = mean_pop
            var_pop_array[i, j]   = var_pop
            
            mean_psi00_array[i, j] = mean_psi00
            var_psi00_array[i, j]  = var_psi00
            
            mean_psi11_array[i, j] = mean_psi11
            var_psi11_array[i, j]  = var_psi11
            
            mean_psi22_array[i, j] = mean_psi22
            var_psi22_array[i, j]  = var_psi22
        end
    end  # Ende der parallelisierten Schleife
    
    ################################################################################
    # Definiere das Ausgabeverzeichnis (das Verzeichnis, in dem das Skript liegt)
    output_dir = @__DIR__
    
    # Erstelle und speichere die Heatmaps mit der analytischen Linie:
    # 1. Für |a| (Kavitätenfeld)
    p_mean_a = heatmap(omega_values, v_values, mean_a_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Mittelwert von |a| (Kavitätenfeld)",
        colorbar_title = "Mittelwert")
    plot!(p_mean_a, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_mean_a)
    savefig(p_mean_a, joinpath(output_dir, sim_label * "_mean_a.png"))
    
    p_var_a = heatmap(omega_values, v_values, var_a_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Varianz von |a| (Kavitätenfeld)",
        colorbar_title = "Varianz")
    plot!(p_var_a, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_var_a)
    savefig(p_var_a, joinpath(output_dir, sim_label * "_var_a.png"))
    
    # 2. Für die Kavitätenpopulation |a|²
    p_mean_pop = heatmap(omega_values, v_values, mean_pop_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Mittelwert der Kavitätenpopulation |a|²",
        colorbar_title = "Mittelwert")
    plot!(p_mean_pop, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_mean_pop)
    savefig(p_mean_pop, joinpath(output_dir, sim_label * "_mean_pop.png"))
    
    p_var_pop = heatmap(omega_values, v_values, var_pop_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Varianz der Kavitätenpopulation |a|²",
        colorbar_title = "Varianz")
    plot!(p_var_pop, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_var_pop)
    savefig(p_var_pop, joinpath(output_dir, sim_label * "_var_pop.png"))
    
    # 3. Für ψ₀₀
    p_mean_psi00 = heatmap(omega_values, v_values, mean_psi00_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Mittelwert von ψ₀₀",
        colorbar_title = "Mittelwert")
    plot!(p_mean_psi00, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_mean_psi00)
    savefig(p_mean_psi00, joinpath(output_dir, sim_label * "_mean_psi00.png"))
    
    p_var_psi00 = heatmap(omega_values, v_values, var_psi00_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Varianz von ψ₀₀",
        colorbar_title = "Varianz")
    plot!(p_var_psi00, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_var_psi00)
    savefig(p_var_psi00, joinpath(output_dir, sim_label * "_var_psi00.png"))
    
    # 4. Für ψ₁₁
    p_mean_psi11 = heatmap(omega_values, v_values, mean_psi11_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Mittelwert von ψ₁₁",
        colorbar_title = "Mittelwert")
    plot!(p_mean_psi11, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_mean_psi11)
    savefig(p_mean_psi11, joinpath(output_dir, sim_label * "_mean_psi11.png"))
    
    p_var_psi11 = heatmap(omega_values, v_values, var_psi11_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Varianz von ψ₁₁",
        colorbar_title = "Varianz")
    plot!(p_var_psi11, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_var_psi11)
    savefig(p_var_psi11, joinpath(output_dir, sim_label * "_var_psi11.png"))
    
    # 5. Für ψ₂₂
    p_mean_psi22 = heatmap(omega_values, v_values, mean_psi22_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Mittelwert von ψ₂₂",
        colorbar_title = "Mittelwert")
    plot!(p_mean_psi22, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_mean_psi22)
    savefig(p_mean_psi22, joinpath(output_dir, sim_label * "_mean_psi22.png"))
    
    p_var_psi22 = heatmap(omega_values, v_values, var_psi22_array,
        xlabel = "Omega",
        ylabel = "V",
        title = "Varianz von ψ₂₂",
        colorbar_title = "Varianz")
    plot!(p_var_psi22, omega_values, line_V, linecolor=:red, lw=2, label="Analytical Boundary")
    display(p_var_psi22)
    savefig(p_var_psi22, joinpath(output_dir, sim_label * "_var_psi22.png"))
    
    ################################################################################
    # Speichere alle Arrays und Eingangsparameter in einer HDF5-Datei im aktuellen Verzeichnis
    h5_filename = joinpath(output_dir, sim_label * "_data.h5")
    h5open(h5_filename, "w") do file
        # Speichere die Basisparameter als Gruppe
        grp = create_group(file, "base_params")
        for (k, v) in pairs(base_params)
            grp[string(k)] = v
        end
        file["tspan"] = [t0, t_end]
        file["omega_values"] = omega_values
        file["v_values"] = v_values
        file["mean_a_array"] = mean_a_array
        file["var_a_array"]  = var_a_array
        file["mean_pop_array"] = mean_pop_array
        file["var_pop_array"]  = var_pop_array
        file["mean_psi00_array"] = mean_psi00_array
        file["var_psi00_array"]  = var_psi00_array
        file["mean_psi11_array"] = mean_psi11_array
        file["var_psi11_array"]  = var_psi11_array
        file["mean_psi22_array"] = mean_psi22_array
        file["var_psi22_array"]  = var_psi22_array
    end
    
    println("Simulation ", sim_label, " abgeschlossen. Ergebnisse in ", h5_filename, " gespeichert.")
end
