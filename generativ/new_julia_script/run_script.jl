################################################################################
# Import required packages
using OrdinaryDiffEq       # For ODE solvers
using Plots                # For plotting heatmaps and GIF creation
using HDF5                 # For saving data in HDF5 files
using Statistics           # For mean and variance computations
using Dates                # For timestamps
using FileIO               # (for file management if needed)

################################################################################
# Create a unique output folder in the same directory as the code
# This function creates a folder with the given base name. If it exists,
# it appends an incrementing number to the folder name.
function create_unique_folder(base_dir::String, base_name::String)
    folder = joinpath(base_dir, base_name)
    counter = 1
    while isdir(folder)
        folder = joinpath(base_dir, base_name * "_" * string(counter))
        counter += 1
    end
    mkpath(folder)
    return folder
end

# Create the output folder (e.g. "results", "results_1", ...)
output_dir = create_unique_folder(@__DIR__, "results")
println("Output folder created: ", output_dir)

################################################################################
# Define the Gamma range parameters:
# Here you can choose the start, stop and step size for Gamma.
gamma_min = 0.0
gamma_max = 3.0
gamma_step = 0.5
gamma_range = collect(gamma_min:gamma_step:gamma_max)

################################################################################
# Define fixed parameters (except Omega, V and Gamma, which will be overwritten)
base_params = (
    kappa  = 1.0,
    gamma  = 1.0,     # used also as g₀ in the analytical boundary expression
    Gamma  = 1.0,     # will be overwritten by gamma_range value
    Omega  = 8.0,     # initial value (will be overwritten in the parameter scan)
    delta1 = 1.0,
    delta2 = 1.0,
    eta    = 1.0,
    V      = -8.0     # initial value (will be overwritten in the parameter scan)
)

################################################################################
# Define time span for the simulation
t0 = 0.0
t_end = 2000.0
tspan = (t0, t_end)

################################################################################
# Define simulation initial conditions (state vector with 11 components)
# Simulation 1: All zeros except ψ₀₀ = 1 (Index 3)
simulations = [
    ("Sim1_psi00", [0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im])
    # Add more simulations if desired.
]

################################################################################
# Define parameter ranges for Omega and V.
omega_values = collect(range(0, stop=8, length=50))   # x-axis values for Omega
v_values     = collect(range(-16.0, stop=-0, length=50))  # y-axis values for V

# Function to compute the analytical V line as a function of Omega:
function compute_v_line(omega_vals, params)
    return -params.delta2/2 * (((omega_vals .* params.kappa) ./ (4 * params.eta * params.gamma)).^2 .+ 1)
end

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
    γ     = params.gamma     # e.g. 1.0 (used as g₀ for the analytical boundary)
    Γ     = params.Gamma     # overwritten by gamma_range value
    Ω     = params.Omega     # overwritten in the parameter scan
    δ₁    = params.delta1    # e.g. 1.0
    δ₂    = params.delta2    # e.g. 1.0
    η     = params.eta       # e.g. 1.0
    V     = params.V         # overwritten in the parameter scan

    # Define the ODE system
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
# Prepare arrays to store plot objects for GIF creation (one for each plot type)
mean_a_plots    = Any[]
mean_psi11_plots = Any[]
var_sum_plots   = Any[]

################################################################################
# Loop over the Gamma range
for gamma_val in gamma_range
    # Update the base parameters with the current Gamma value
    current_params_base = merge(base_params, (Gamma = gamma_val,))
    
    # Compute the analytical V line for the current parameters
    current_v_line = compute_v_line(omega_values, current_params_base)
    
    # Loop over the different simulations (here only one is defined)
    for (sim_label, y0) in simulations
        # Append the Gamma value to the simulation label for uniqueness
        new_sim_label = sim_label * "_Gamma_" * string(gamma_val)
        println("Starting simulation: ", new_sim_label)
        
        n_omega = length(omega_values)
        n_v = length(v_values)
        
        # Pre-allocate arrays for the mean and variance of observables:
        mean_a_array     = zeros(n_v, n_omega)
        var_a_array      = zeros(n_v, n_omega)
        mean_pop_array   = zeros(n_v, n_omega)
        var_pop_array    = zeros(n_v, n_omega)
        mean_psi00_array = zeros(n_v, n_omega)
        var_psi00_array  = zeros(n_v, n_omega)
        mean_psi11_array = zeros(n_v, n_omega)
        var_psi11_array  = zeros(n_v, n_omega)
        mean_psi22_array = zeros(n_v, n_omega)
        var_psi22_array  = zeros(n_v, n_omega)
        
        for (i, v) in enumerate(v_values)
            for (j, omega) in enumerate(omega_values)
                println("Processing Omega = ", omega, ", V = ", v)
                # Update parameters for the current run (Omega and V are overwritten)
                current_params = merge(current_params_base, (Omega = omega, V = v))
                prob = ODEProblem((y, p, t) -> rhs_dgl(y, p), y0, tspan, current_params)
                sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8)
                
                # Use the last 25% of the time series for evaluation:
                n_t = length(sol.t)
                start_index = floor(Int, 0.75 * n_t) + 1
                
                # Extract observables from the last quarter of the time series:
                last_quarter_a    = sol[1, start_index:end]
                last_quarter_pop  = abs.(sol[1, start_index:end]).^2
                last_quarter_psi00 = real.(sol[3, start_index:end])
                last_quarter_psi11 = real.(sol[6, start_index:end])
                last_quarter_psi22 = real.(sol[7, start_index:end])
                
                # Compute means and variances
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
                
                # Store the computed values in the arrays
                mean_a_array[i, j]     = mean_a
                var_a_array[i, j]      = var_a
                mean_pop_array[i, j]   = mean_pop
                var_pop_array[i, j]    = var_pop
                mean_psi00_array[i, j] = mean_psi00
                var_psi00_array[i, j]  = var_psi00
                mean_psi11_array[i, j] = mean_psi11
                var_psi11_array[i, j]  = var_psi11
                mean_psi22_array[i, j] = mean_psi22
                var_psi22_array[i, j]  = var_psi22
            end
        end
        
        # Define the output HDF5 filename for this Gamma value
        h5_filename = joinpath(output_dir, new_sim_label * "_data.h5")
        
        # Save the data into the HDF5 file
        h5open(h5_filename, "w") do file
            write(file, "omega_values", omega_values)
            write(file, "v_values", v_values)
            write(file, "mean_a_array", mean_a_array)
            write(file, "var_a_array", var_a_array)
            write(file, "mean_pop_array", mean_pop_array)
            write(file, "var_pop_array", var_pop_array)
            write(file, "mean_psi00_array", mean_psi00_array)
            write(file, "var_psi00_array", var_psi00_array)
            write(file, "mean_psi11_array", mean_psi11_array)
            write(file, "var_psi11_array", var_psi11_array)
            write(file, "mean_psi22_array", mean_psi22_array)
            write(file, "var_psi22_array", var_psi22_array)
            write(file, "tspan", [t0, t_end])
        end
        
        ################################################################################
        # Load data from the HDF5 file and generate separate plots.
        h5open(h5_filename, "r") do file
            merged_omega = read(file, "omega_values")
            merged_v = read(file, "v_values")
            # Recompute the analytical V line for the merged Omega grid:
            merged_v_line = compute_v_line(merged_omega, current_params_base)
            x_limits = (minimum(merged_omega), maximum(merged_omega))
            y_limits = (minimum(merged_v), maximum(merged_v))
            
            # Plot 1: Combined Mean of |a| (Cavity Field)
            merged_mean_a = read(file, "mean_a_array")
            p1 = heatmap(merged_omega, merged_v, merged_mean_a,
                xlabel = "Omega",
                ylabel = "V",
                title = "Combined Mean of |a| (Cavity Field) (Gamma = $(gamma_val))",
                colorbar_title = "Mean",
                xlims = x_limits,
                ylims = y_limits)
            plot!(p1, merged_omega, merged_v_line, linecolor=:red, lw=2, label="Analytical V line")
            savefig(p1, joinpath(output_dir, new_sim_label * "_mean_a.png"))
            push!(mean_a_plots, p1)
            
            # Plot 2: Combined Mean of ψ₁₁
            merged_mean_psi11 = read(file, "mean_psi11_array")
            p2 = heatmap(merged_omega, merged_v, merged_mean_psi11,
                xlabel = "Omega",
                ylabel = "V",
                title = "Combined Mean of ψ₁₁ (Gamma = $(gamma_val))",
                colorbar_title = "Mean",
                xlims = x_limits,
                ylims = y_limits)
            plot!(p2, merged_omega, merged_v_line, linecolor=:red, lw=2, label="Analytical V line")
            savefig(p2, joinpath(output_dir, new_sim_label * "_mean_psi11.png"))
            push!(mean_psi11_plots, p2)
            
            # Plot 3: Sum of Variances (ψ₀₀, ψ₁₁, ψ₂₂)
            merged_var_psi00 = read(file, "var_psi00_array")
            merged_var_psi11 = read(file, "var_psi11_array")
            merged_var_psi22 = read(file, "var_psi22_array")
            merged_var_sum = merged_var_psi00 .+ merged_var_psi11 .+ merged_var_psi22
            p3 = heatmap(merged_omega, merged_v, merged_var_sum,
                xlabel = "Omega",
                ylabel = "V",
                title = "Sum of Variances (ψ₀₀, ψ₁₁, ψ₂₂) (Gamma = $(gamma_val))",
                colorbar_title = "Variance Sum",
                xlims = x_limits,
                ylims = y_limits)
            plot!(p3, merged_omega, merged_v_line, linecolor=:red, lw=2, label="Analytical V line")
            savefig(p3, joinpath(output_dir, new_sim_label * "_variance_sum.png"))
            push!(var_sum_plots, p3)
        end
        
        println("Simulation ", new_sim_label, " completed. Data and plots saved.")
    end
end

################################################################################
# Create GIFs from the accumulated plots (one GIF per plot type)

println("All simulations completed.")
