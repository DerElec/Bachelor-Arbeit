using OrdinaryDiffEq
using Plots

# Define the ODE function (English comments)
function rhs_dgl(y, params)
    # Unpack state vector
    a         = y[1]
    a_dagger  = y[2]
    psi00     = y[3]
    psi01     = y[4]
    psi10     = y[5]
    psi11     = y[6]
    psi22     = y[7]
    psi21     = y[8]
    psi12     = y[9]
    psi20     = y[10]
    psi02     = y[11]

    # Unpack parameters
    kappa   = params[:kappa]
    gamma   = params[:gamma]
    Gamma   = params[:Gamma]
    Omega   = params[:Omega]
    delta1  = params[:delta1]
    delta2  = params[:delta2]
    eta     = params[:eta]
    V       = params[:V]

    # ODE equations
    da_dt = -kappa/2 * a - 1im*(gamma*psi01) + eta
    da_dagger_dt = conj(da_dt)

    dpsi00_dt = Gamma*psi11 + 1im*gamma*(psi10*a - psi01*a_dagger)
    dpsi01_dt = -Gamma/2 * psi01 + 1im*(-delta1*psi01 + gamma*(psi11*a - psi00*a) - Omega/2*psi02)
    dpsi10_dt = conj(dpsi01_dt)

    dpsi11_dt = -Gamma*psi11 + 1im*gamma*(psi01*a_dagger - psi10*a) + 1im*(Omega/2)*(psi21 - psi12)
    dpsi22_dt = 1im*(Omega/2)*(psi12 - psi21)

    dpsi21_dt = -Gamma/2 * psi21 + 1im*(delta2*psi21 - delta1*psi21 - gamma*psi20*a + (Omega/2)*(psi11 - psi22) + 2*V*psi21*psi22)
    dpsi12_dt = conj(dpsi21_dt)

    dpsi02_dt = 1im*(-delta2*psi02 - Omega/2*psi01 - 2*V*psi02*psi22 + gamma*psi12*a)
    dpsi20_dt = conj(dpsi02_dt)

    return [da_dt, da_dagger_dt, dpsi00_dt, dpsi01_dt, dpsi10_dt, dpsi11_dt, dpsi22_dt, dpsi21_dt, dpsi12_dt, dpsi20_dt, dpsi02_dt]
end

# Helper function: Trapezoidal integration over a (possibly non-uniform) grid
# Function to compute the integral using the trapezoidal rule
function trapz(x::Vector{<:Real}, y::Vector{<:Real})
    integral = 0.0
    for i in 1:(length(x)-1)
        dx = x[i+1] - x[i]               # Difference between consecutive x values
        integral += (y[i] + y[i+1]) * dx / 2  # Trapezoidal area
    end
    return integral
end

# Function to compute the time average and variance over the last quarter of the simulation time
function computeTimeStats(imbalance::Vector{<:Real}, tvals::Vector{<:Real})
    # Determine the last quarter of the simulation time
    T_end = tvals[end]
    T_start_window = T_end - (T_end - tvals[1]) / 4
    indices = findall(t -> t >= T_start_window, tvals)
    t_window = tvals[indices]
    imbalance_window = imbalance[indices]
    
    # Compute the total time interval in the window
    ΔT = t_window[end] - t_window[1]
    
    # Compute the time average using the trapezoidal rule
    avg_imbalance = trapz(t_window, imbalance_window) / ΔT
    
    # Compute the variance using the trapezoidal rule
    variance_imbalance = trapz(t_window, (imbalance_window .- avg_imbalance).^2) / ΔT
    
    return avg_imbalance, variance_imbalance
end
# Flag to choose simulation mode:
# Set run_six_simulations = true to perform all 6 simulation variants.
# Standardmäßig (run_six_simulations = false) wird nur psi00=1 und psi22=1 verwendet.
run_six_simulations = false

# Default simulation initial conditions: bosonic mode and atomic states.
# a and a_dagger are computed as 2*eta/kappa (mit eta=1 und kappa=1 => a=2)
a0 = 0.0 + 0im
a_dagger0 = 0.0 + 0im
# Default: psi00 = 1, psi22 = 1, alle anderen 0
default_y0 = [a0, a_dagger0, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]

# Base parameters (parameters other than V and Omega, which will be varied)
base_params = (kappa = 1.0,
               gamma = 1.0,
               Gamma = 2.0,
               Omega = 7.0,    # will be overwritten in the sweep
               delta1 = -1.0,
               delta2 = -1.0,
               eta = 1.0,
               V = -6.0)       # will be overwritten in the sweep

# Optional: Run the six simulation variants if desired
if run_six_simulations
    simulations = [
        # 1. Only psi00 = 1
        ("Sim 1: ψ₀₀ = 1", [a0, a_dagger0, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
        # 2. Only psi11 = 1
        ("Sim 2: ψ₁₁ = 1", [a0, a_dagger0, 0.0+0im, 0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
        # 3. Only psi22 = 1
        ("Sim 3: ψ₂₂ = 1", [a0, a_dagger0, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 1.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
        # 4. psi00 = 1/2 and psi11 = 1/2
        ("Sim 4: ψ₀₀ = 0.5, ψ₁₁ = 0.5", [a0, a_dagger0, 0.5+0im, 0.0+0im, 0.0+0im, 0.5+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
        # 5. psi00 = 1/2 and psi22 = 1/2
        ("Sim 5: ψ₀₀ = 0.5, ψ₂₂ = 0.5", [a0, a_dagger0, 0.5+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.5+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im]),
        # 6. psi11 = 1/2 and psi22 = 1/2
        ("Sim 6: ψ₁₁ = 0.5, ψ₂₂ = 0.5", [a0, a_dagger0, 0.0+0im, 0.0+0im, 0.0+0im, 0.5+0im, 0.5+0im, 0.0+0im, 0.0+0im, 0.0+0im, 0.0+0im])
    ]
    
    for (sim_label, y0) in simulations
        println("Running ", sim_label)
        prob = ODEProblem((y, p, t) -> rhs_dgl(y, base_params), y0, (0.0, 2000.0))
        sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8)
        println("Simulation ", sim_label, " abgeschlossen.")
    end
end

# ------------------------------
# Parameter Sweep: Loop over V and Omega
# ------------------------------
Vs = 0:1:8
Omegas = 0:1:8

# Matrices to store the time-averaged imbalance and the time variance
avg_matrix = zeros(length(Vs), length(Omegas))
var_matrix = zeros(length(Vs), length(Omegas))

# Define simulation time span for the sweep
tspan = (0.0, 2000.0)

for (i, V_val) in enumerate(Vs)
    for (j, Omega_val) in enumerate(Omegas)
        # Update parameters for current simulation (only V and Omega are varied)
        params_current = (kappa   = base_params.kappa,
                          gamma   = base_params.gamma,
                          Gamma   = base_params.Gamma,
                          Omega   = Omega_val,
                          delta1  = base_params.delta1,
                          delta2  = base_params.delta2,
                          eta     = base_params.eta,
                          V       = V_val)
        
        # Run simulation with default initial condition (psi00 = 1, psi22 = 1)
        prob = ODEProblem((y, p, t) -> rhs_dgl(y, params_current), default_y0, tspan)
        sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8)
        tvals = sol.t
        
        # Extract psi00 (index 3) and psi22 (index 7)
        psi00_vals = real.(sol[3, :])
        psi22_vals = real.(sol[7, :])


        a_vals = [psi00_vals]
        
        # Compute population imbalance: psi22 - psi00
        imbalance = psi22_vals .- psi00_vals
        
        avg_imbalance, variance_imbalance=computeTimeStats(imbalance, tvals)
        avg_a, variance_a =computeTimeStats(imbalance, tvals)
        avg_psi00, variance_psi00=computeTimeStats(psi00_vals, tvals)
        
        # Store the computed values in the matrices
        avg_matrix[i, j] = avg_imbalance
        var_matrix[i, j] = variance_imbalance
    end
end


funct




avg_heatmap = heatmap(Omegas, Vs, avg_matrix, xlabel="Omega", ylabel="V",
                      title="Zeitdurchschnittliche Populationsimbalance", colorbar_title="Avg Imbalance")
display(avg_heatmap)

var_heatmap = heatmap(Omegas, Vs, var_matrix, xlabel="Omega", ylabel="V",
                      title="Zeitvarianz der Populationsimbalance", colorbar_title="Varianz")
display(var_heatmap)