using Random
using Distributions
using Distances
using SimulatedAnnealingABC
using Pkg
Pkg.add("Plots")
using Plots

Random.seed!(1111)					  

prior = Uniform(-1,1)

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

function f_dist(param)
    # Define the distributions
    dist1 = Normal(param, 0.1)
    dist_others = Normal(0, 1)

    # Draw from the distributions
    y1 = rand(dist1)
    y_others = rand(dist_others, 9)

    # Calculate the components of the return vector
    comp1 = abs(y1) / (0.01 + abs(y1))
    comp_others = abs.(y_others)

    # Combine the components into a 10d vector
    return [comp1; comp_others]
end

# --- TYPE 1 -> single-ϵ ---
out_single_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 1)

# --- TYPE 2 -> multi-ϵ ---
out_multi_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 10.0, type = 2)

# --- TYPE 3 -> hybrid multi-u-single-ϵ ---
out_hybrid = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 3)

out_single_eps_2 = update_population!(out_single_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 1)
out_multi_eps_2 = update_population!(out_multi_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 2)

# Sample data: a vector of 1000 10-dimensional vectors
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10
data = out_multi_eps.state.ρ_history
matrix_data_multi = hcat(data...)'  # Transpose after concatenation to get 1000x10
data = out_hybrid.state.ρ_history
matrix_data_hybrid = hcat(data...)'  # Transpose after concatenation to get 1000x10


iter = size(data)[1]

plot1=plot(1:iter, matrix_data_single, color="red")
plot!(plot1,1:iter, matrix_data_multi, color="blue")
plot!(plot1,1:iter, matrix_data_hybrid, color="green")

true_sample = rand(Normal(0, 0.1), 1000)

hist1=histogram(out_single_eps.population,bins=30,color="red")
histogram!(hist1,out_multi_eps.population,bins=30,color="blue")
histogram!(hist1,out_hybrid.population,bins=30,color="green")
histogram!(hist1,true_sample,bins=30,color="yellow")
