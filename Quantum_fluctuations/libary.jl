using QuantumCumulants
using ModelingToolkit, OrdinaryDiffEq
#using Plots 
using QuantumOptics
using CGcoefficient


    #correct
function tupel_existiert_unordered(arr, tup)
    return any(t -> (t[1] == tup[1] && t[2] == tup[2]) || (t[1] == tup[2] && t[2] == tup[1]), arr)

end

function tupel_existiert_ordered(arr, tup)
        # Überprüft, ob das geordnete Tupel existiert
    return any(t -> t[1] == tup[1] && t[2] == tup[2], arr)
end


function create_u0(u1, u2, l,ground_states,complex)
    # Initialisieren des u0-Vektors mit komplexen Nullen
    if complex
        u0 = zeros(ComplexF64, l)
    else
        u0 = zeros(Float64, l)
    end

    # Zuweisen der initialen Zustände
    u0[1] = u1
    u0[2] = u2
    if complex 
            u0[3]= ground_states[1]+0im
            u0[4]= ground_states[2]+0im
            u0[5]= ground_states[3]+0im
            u0[6]= ground_states[4]+0im
            u0[7]= ground_states[5]+0im
        else 
            u0[3]= ground_states[1]
            u0[4]= ground_states[2]
            u0[5]= ground_states[3]
            u0[6]= ground_states[4]
            u0[7]= ground_states[5]
    end



    println("Initialer Grundzustand: ", u0)

    return u0
end

function create_atomic_levels_symbolic(F::Int, state_label::Symbol)
    return [(state_label, m) for m in collect(-F:F)]
end


#creates array of sublevels
function create_atomic_array(F::Int)
    return collect(-F:F)
end


using Symbolics
# function clebsch_gordan(m::Int, mp::Int, q::Int)::Float64
#     # Definieren der Clebsch-Gordan-Koeffizienten als Dictionary
#     # Schlüssel sind Tupel (m, mp), Werte sind die entsprechenden Koeffizienten
#     coefficients = Dict(
#         (-2, -1) => sqrt(1/7),    # C(-2 -> -1)
#         (-1,  0) => sqrt(2/7),    # C(-1 -> 0)
#         (-1, -2) => sqrt(5/7),    # C(-1 -> -2)
#         ( 0,  1) => sqrt(3/7),    # C(0 -> 1)
#         ( 0,  0) => sqrt(4/7),    # C(0 -> 0)
#         ( 0, -1) => 0.0,           # C(0 -> -1) nicht erlaubt
#         ( 1,  2) => sqrt(3/7),    # C(1 -> 2)
#         ( 1,  1) => sqrt(4/7),    # C(1 -> 1)
#         ( 1,  0) => 0.0,           # C(1 -> 0) nicht erlaubt
#         ( 2,  3) => sqrt(1/7),    # C(2 -> 3)
#         ( 2,  2) => sqrt(2/7),    # C(2 -> 2)
#         ( 2,  1) => sqrt(4/7)     # C(2 -> 1)
#     )

# Schlüssel: (m1, q)
cg_coefficients = Dict(
    (-2, -1) => sqrt(1/2),#sqrt(1/15),
    (-2, 0)  => 0,#0,
    (-2, 1)  => sqrt(1/30),#sqrt(1/15),
    (-1, -1) => sqrt(1/3),#sqrt(2/15),
    (-1, 0)  => 0,#sqrt(1/5),
    (-1, 1)  => sqrt(1/10),#sqrt(2/15),
    (0, -1)  => sqrt(1/5),#sqrt(1/5),
    (0, 0)   => 0,#0,
    (0, 1)   => sqrt(1/5),#sqrt(1/5),
    (1, -1)  => sqrt(1/10),#sqrt(2/15),
    (1, 0)   => 0,#0,
    (1, 1)   => sqrt(1/3),#sqrt(2/15),
    (2, -1)  => sqrt(1/30),#sqrt(1/15),
    (2, 0)   => 0,#0,
    (2, 1)   => sqrt(1/2)#sqrt(1/15)
)

function generate_system(u1,u2,ground_states,complex,symbolic_eq)

    global hbar,kappa,kappa_p,kappa_m, Gamma,delta_p,delta_a,delta_b, eta_p,eta_m,N,g0= cnumbers("ℏ κ κ_p κ_m Γ δ A B η_p η_m N g0")

    # Definieren Sie die benötigten globalen Variablen
    I = 3/2       # Kernspin von Rubidium 87
    Jg = 1/2      # Gesamtdrehimpuls des Grundzustands (5S₁/₂)
    Je = 3/2      # Gesamtdrehimpuls des angeregten Zustands (5P₃/₂)
    F = 2        # Hyperfein-Zustand des Grundzustands
    F_prime =3   # Hyperfein-Zustand des angeregten Zustands


    ground_levels=create_atomic_array(F)           
    excited_levels=create_atomic_array(F_prime)    
    levels = vcat(ground_levels, excited_levels)

    ground_levels_symbolic = create_atomic_levels_symbolic(F, :g)
    excited_levels_symbolic = create_atomic_levels_symbolic(F_prime, :e)

    levels_symbolic = vcat(ground_levels_symbolic, excited_levels_symbolic)


    # Define Hilbert spaces without labels
    h_cav_p = FockSpace(:cavity_p)  # Hilbert space for mode a₊
    h_cav_m = FockSpace(:cavity_m)  # Hilbert space for mode a₋
    #h_atom = NLevelSpace(:atom,levels_symbolic)
    
    h_atom = NLevelSpace(:atom,((:e, 3),(:g, -2),(:g, -1),(:g, 0),(:g, 1),(:g, 2),(:e,-3),(:e, -2),(:e, -1), (:e, 0),(:e, 1),
    (:e, 2),))

    h = tensor(h_cav_p,h_cav_m,h_atom)

    sigma(α, β, k) = IndexedOperator(Transition(h,:σ,α,β,3), k)

    g(m,mp) = IndexedVariable(:g, m,mp)
    branching_symb(m,mp) = IndexedVariable(:b, m,mp)
    j = Index(h,:j, N, h_atom)
    k = Index(h,:k, N, h_atom)

    @qnumbers a::Destroy(h,1)
    @qnumbers b::Destroy(h,2)

    
    # Define the dictionary to hold the values
    global branching_ratios= Dict{Tuple{Int, Int}, Float64}()

    # Populate the dictionary with the values from the branching_ratios
    # Row -3
    branching_ratios[(-3, -2)] = 1.0
    branching_ratios[(-3, -1)] = 0.0
    branching_ratios[(-3, 0)]  = 0.0
    branching_ratios[(-3, 1)]  = 0.0
    branching_ratios[(-3, 2)]  = 0.0

    # Row -2
    branching_ratios[(-2, -2)] = 1/3
    branching_ratios[(-2, -1)] = 2/3
    branching_ratios[(-2, 0)]  = 0.0
    branching_ratios[(-2, 1)]  = 0.0
    branching_ratios[(-2, 2)]  = 0.0

    # Row -1
    branching_ratios[(-1, -2)] = 1/15
    branching_ratios[(-1, -1)] = 8/15
    branching_ratios[(-1, 0)]  = 6/15
    branching_ratios[(-1, 1)]  = 0.0
    branching_ratios[(-1, 2)]  = 0.0

    # Row 0
    branching_ratios[(0, -2)]  = 0.0
    branching_ratios[(0, -1)]  = 1/5
    branching_ratios[(0, 0)]   = 3/5
    branching_ratios[(0, 1)]   = 1/5
    branching_ratios[(0, 2)]   = 0.0

    # Row 1
    branching_ratios[(1, -2)]  = 0.0
    branching_ratios[(1, -1)]  = 0.0
    branching_ratios[(1, 0)]   = 6/15
    branching_ratios[(1, 1)]   = 8/15
    branching_ratios[(1, 2)]   = 1/15

    # Row 2
    branching_ratios[(2, -2)]  = 0.0
    branching_ratios[(2, -1)]  = 0.0
    branching_ratios[(2, 0)]   = 0.0
    branching_ratios[(2, 1)]   = 2/3
    branching_ratios[(2, 2)]   = 1/3

    # Row 3
    branching_ratios[(3, -2)]  = 0.0
    branching_ratios[(3, -1)]  = 0.0
    branching_ratios[(3, 0)]   = 0.0
    branching_ratios[(3, 1)]   = 0.0
    branching_ratios[(3, 2)]   = 1.0

    


    global H_cav = -hbar *delta_p*(a'*a+b'*b)

    global H_pump= im*eta_p*(a'-a) +im*eta_m*(b'-b) #hbar *im*(eta_p*(a'-a) +eta_m*(b'-b))

    #global Ha_atom=0; # problem - if 
    # for m in ground_levels
    #     # if m <1
    #     #     print("catch")
    #     # else
    #     #     mp=m+1
    #     #     global Ha_atom-=hbar*delta_p*Σ(sigma((:e, mp), (:g, m),k)*sigma((:g, m), (:e, mp),k),k)
    #     # end
    #     mp=m+1
    #     global Ha_atom-=hbar*delta_p*Σ(sigma((:e, mp), (:g, m),k)*sigma((:g, m), (:e, mp),k),k)
    #     mp=m-1
    #     #global Ha_atom-=hbar*delta_p*Σ(sigma((:g, m), (:e, mp),k)'*sigma((:g, m), (:e, mp),k),k)
    #     global Ha_atom-=hbar*delta_p*Σ(sigma((:e, mp), (:g, m),k)'*sigma((:g, m), (:e, mp),k),k)
    # end
    global Ha_atom=-delta_p*Σ((sigma((:e, -3),(:g, -2),k)*sigma((:g, -2),(:e, -3),k)+sigma((:e, -2),(:g, -1),k)*sigma((:g, -1),(:e, -2),k)+sigma((:e, -1),(:g, 0),k)*sigma((:g, 0),(:e, -1),k)+sigma((:e, 0),(:g, 1),k)*sigma((:g, 1),(:e, 0),k)+sigma((:e, 1),(:g, 2),k)*sigma((:g, 2),(:e, 1),k)+sigma((:e, 2),(:g, 1),k)*sigma((:g, 1),(:e, 2),k)+sigma((:e, 3),(:g, 2),k)*sigma((:g, 2),(:e, 3),k)),k)
    H_0=H_cav+H_pump+Ha_atom;


    global H_int = 0
    for m in ground_levels
        mp = m + 1
        if mp in excited_levels
            if abs(m-mp) ==0
                global H_int +=0
            else 
                if symbolic_eq
                    global H_int += 1im * hbar * Σ(a' * sigma((:g, m), (:e, mp),k) - a * sigma((:g, m), (:e, mp),k)',k)*g(m,mp)
                else
                    global H_int += 1im * hbar * Σ(a' * sigma((:g, m), (:e, mp),k) - a * sigma((:g, m), (:e, mp),k)',k)*g0*cg_coefficients[(m,1)]
                end
                    #*CG(F,m,F_prime,mp,F_prime,mp)#*calc_clebsch_gordan(m,mp,1,F,F_prime)*g0#*g0*sqrt(N)
            end
        end
        mp = m - 1
        if mp in excited_levels
            if abs(m-mp) ==0
                global H_int +=0
            else
                if symbolic_eq
                    global H_int += 1im * hbar * Σ(b' * sigma((:g, m), (:e, mp),k) - b * sigma((:g, m), (:e, mp),k)',k)*g(m,mp)
                else
                    global H_int += 1im * hbar * Σ(b' * sigma((:g, m), (:e, mp),k) - b * sigma((:g, m), (:e, mp),k)',k)*g0*cg_coefficients[(m,-1)]#*CG(F,m,F_prime,mp,F_prime,mp)#*calc_clebsch_gordan(m,mp,-1,F,F_prime)*g0#*g0*sqrt(N)
                end
            end
        end
    end
    H = H_0+H_int;
    println(H)
    L_cav=[a,b];
    rates_cav=[2*kappa_p,2*kappa_m];

    L_at= [];
    rates_at=[];

    for a in excited_levels
        for b in ground_levels
            if (abs(a-b)==1 || abs(a-b)==0)
                push!(L_at,sigma((:g, b), (:e, a),k))
                if symbolic_eq
                    push!(rates_at,Gamma*branching_symb(a,b))
                else
                    push!(rates_at,Gamma*branching_ratios[a,b])
                end
            end
        end
    end




    #ops_cav=[a,b,a'a,b'b]
    ops_cav=[a,b]
    ops_at=[]
    known_transitions_ee = []
    known_transitions_gg = []
    known_transitions_ge = []
    known_transitions_eg = []

    global u0=Complex{Float64}[0.0 + 0im ,0.0 + 0im ]

    #transitions from excited to excited
    for e1 in excited_levels
        for e2 in excited_levels
            if (abs(e1-e2)==0 || iseven(abs(e1-e2))) && !tupel_existiert_unordered(known_transitions_ee, [e1, e2]) 
                push!(ops_at,sigma((:e, e1), (:e, e2),j))
                push!(known_transitions_ee,[e1,e2])

                if e1==e2 && e1==2
                    push!(u0,1.0 + 0im)
                else
                    push!(u0,0.0 + 0im)
                end
            end
        end
    end

    for e1 in ground_levels
        for e2 in ground_levels
            if (abs(e1-e2)==0 || iseven(abs(e1-e2))) && !tupel_existiert_unordered(known_transitions_gg, [e1, e2]) 
                if e1==e2 && e2==-2
                    print("catch")
                else
                    push!(ops_at,sigma((:g, e1), (:g, e2),j))
                    push!(known_transitions_gg,[e1,e2])
                    push!(u0,0.0 + 0im)
                end
                
            end
        end
    end

    for e1 in ground_levels
        for e2 in excited_levels
            if  isodd(abs(e1-e2))  && !tupel_existiert_ordered(known_transitions_ge, [e1, e2]) 
                push!(ops_at,sigma((:e, e2), (:g, e1),j))
                push!(known_transitions_ge,[e1,e2])
                
                push!(u0,0.0 + 0im)
                #test_g=e1+3
                #test_e=e2+4
                #println("sigma_g$test_g e$test_e")
            end    

        end
    end





    L=vcat(L_cav,L_at);
    rates=vcat(rates_cav,rates_at);


    
    ops=[a,b,sigma((:g, -2), (:g, -2),j),sigma((:g, -1), (:g, -1),j),sigma((:g, 0), (:g, 0),j),sigma((:g, 1), (:g, 1),j),sigma((:g, 2), (:g, 2),j),sigma((:e, -3), (:e, -3),j),sigma((:e, -2), (:e, -2),j),sigma((:e, -1), (:e, -1),j),sigma((:e, 0), (:e, 0),j),sigma((:e, 1), (:e, 1),j),sigma((:e, 2), (:e, 2),j)]
    eqs_1 = meanfield(ops,H,L;rates=rates,order=1); #working

    
    
    eqs_c = complete(eqs_1) #complete
   
    eqs_sc = QuantumCumulants.scale(eqs_c) #scaled
    eqs =QuantumCumulants.simplify(eqs_sc) #simplified
    @named sys = ODESystem(eqs)
    
    l = length(u0)
    u0_random = create_u0(u1, u2, l,ground_states,complex)

    return eqs,sys,u0_random,L,rates,ops
    ############################################### until first sim 
end












# function create_gif_animation(all_times,all_deltas,heatmap_data,path)
#     # Define x, y, and z for plotting
#     x = all_times
#     y = all_deltas
#     z = heatmap_data

#     # Number of frames in the animation
#     n_frames = 60  # Adjust for smoother animation

#     # Create the animation
#     anim = @animate for i in range(0, stop = 2π, length = n_frames)
#         # Create the surface plot
#         p = surface(
#             x,
#             y,
#             z,
#             xlabel = "Time (s)",
#             ylabel = "Delta Values (Δ)",
#             zlabel = "Intensity |a|²",
#             title = "Simulation Results",
#             colorbar_title = "Intensity |a|²",
#             c = :viridis,  # Choose a color gradient
#         )

#         # Adjust the camera angle to rotate around the plot
#         azimuth = 360 * (i / (2π))  # Convert radians to degrees
#         elevation = 30              # Fixed elevation angle
#         plot!(p, camera = (azimuth, elevation))

#         # Optionally, add other plot elements or annotations here
#     end

#     # Save the animation as a GIF
#     gif(anim, path, fps = 15)

#     # Display the animation (if supported)
#     display(anim)

# end


function create_3d_plot(all_times, all_deltas, heatmap_data)
    # Define x, y, and z for plotting
    x = all_times
    y = all_deltas
    z = heatmap_data

    # Create the 3D surface plot
    p = surface(
        x,
        y,
        z,
        xlabel = "Time (s)",
        ylabel = "Delta Values (Δ)",
        zlabel = "Intensity |a|²",
        title = "Simulation Results",
        colorbar_title = "Intensity |a|²",
        c = :viridis  # Choose a color gradient
    )

    # Set an initial camera angle for better visualization
    azimuth = 30   # Azimuthal rotation angle
    elevation = 30 # Elevation angle
    plot!(p, camera = (azimuth, elevation))

    # Display the plot
    display(p)
end
# Angepasste Funktion verwenden
function plot_delta_slice(delta_value, all_times, all_deltas, heatmap_data; component_index=1)
    # Prüfen, ob die Dimensionen korrekt sind
    if size(heatmap_data) == (length(all_deltas), length(all_times))
        # Transponieren der Matrix für korrekte Ausrichtung
        heatmap_data = heatmap_data'
    elseif size(heatmap_data) != (length(all_times), length(all_deltas))
        error("Dimension mismatch: heatmap_data must have size (length(all_times), length(all_deltas)).")
    end

    # Find the index of the closest delta value
    delta_index = findfirst(x -> isapprox(x, delta_value, atol=1e-6), all_deltas)

    # Error handling if delta_value is not in all_deltas
    if delta_index === nothing
        error("Delta value $delta_value not found in all_deltas.")
    end

    # Extract the slice corresponding to the given delta value
    slice_data = heatmap_data[:, delta_index]

    # Create the 2D plot
    p = plot(
        all_times,
        slice_data,
        xlabel = "Time (s)",
        ylabel = "Intensity |component $component_index|²",
        title = "2D Slice for Δ = $delta_value, Component = $component_index",
        label = "Δ = $delta_value",
        lw = 2,
        color = :blue
    )

    # Display the plot
    display(p)
end


function simulate_dynamics(start_t_p,end_sim_t,parameterset_1,parameterset_2,sys,u0_random)

    ps = [hbar, kappa, N ,kappa_p,kappa_m,g0,Gamma,delta_p,eta_p,eta_m]
    Gamma_1=parameterset_1[7]
    p0 = parameterset_1
    prob = ODEProblem(sys,u0_random,(0.0, start_t_p*Gamma_1), ps.=>p0)
    sol1 = solve(prob,Tsit5(), abstol=1e-8, reltol=1e-10)

    u0_new=sol1.u[end]
    Gamma_2=parameterset_2[7]

    ps_new = [hbar, kappa, N ,kappa_p,kappa_m,g0,Gamma,delta_p,eta_p,eta_m]
    p0_new = parameterset_2

    prob_2 = ODEProblem(sys,u0_new,(0.0, end_sim_t*Gamma_2), ps_new.=>p0_new)
    sol2 = solve(prob_2,Tsit5());

    shifted_t = sol2.t .+ sol1.t[end]
    shifted_t = shifted_t[2:end]  # Erstes Element entfernen

    # Zeiten kombinieren
    times = vcat(sol1.t, shifted_t)

    # Werte kombinieren
    values = vcat(sol1.u, sol2.u[2:end])  # Kombinieren der Vektoren


    return times,values

end

