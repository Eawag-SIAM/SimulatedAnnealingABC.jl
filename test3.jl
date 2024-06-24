using Random
using Distributions
using Distances
using SimulatedAnnealingABC
using Plots


function f_dist(param)
    # Draw from the distribution
    y = rand(Normal(param,0.1),1)[1]
    s=y^2

    y_others = rand(Normal(0, 1), 9)
    s_others = abs.(y_others)

    return vcat(s,s_others)
end

function posterior_sample(n_samples=1000)
    
    rand(Normal(0,0.1), n_samples)
end

Random.seed!(1111)					  

# Set parameters

prior = Uniform(-1,1)

np = 1000       # number of particles
ns = 5_000_000  # number of particle updates

# synthetic data and summary stats:
# yobs = sample_mixture(Normal(param_true,1),Normal(-param_true,sigma),mixing_coeff,2)


# true (approx) posterior
samples = posterior_sample(np)
histogram(samples,bins=100)

# --- TYPE 1 -> single-ϵ ---
out_single_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 1)
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10

out_single_eps = update_population!(out_single_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 1)
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10

# --- TYPE 2 -> multi-ϵ ---
out_multi_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 0.05, type = 2)
data = out_multi_eps.state.ρ_history
matrix_data_multi = hcat(data...)'  # Transpose after concatenation to get 1000x10

out_multi_eps = update_population!(out_multi_eps, f_dist, prior; n_simulation = ns, v = 0.01, type = 2)
data = out_multi_eps.state.ρ_history
matrix_data_multi = hcat(data...)'  # Transpose after concatenation to get 1000x10

# --- TYPE 3 -> hybrid multi-u-single-ϵ ---
out_hybrid = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 3)
data = out_hybrid.state.ρ_history
matrix_data_hybrid = hcat(data...)'  # Transpose after concatenation to get 1000x10

out_hybrid = update_population!(out_hybrid, f_dist, prior; n_simulation = ns, v = 1.0, type = 3)
data = out_hybrid.state.ρ_history
matrix_data_hybrid = hcat(data...)'  # Transpose after concatenation to get 1000x10

iter = size(data)[1]
plot1=plot(1:iter, matrix_data_single,  yscale=:log10, color="red")
plot!(plot1,1:iter, matrix_data_multi,  yscale=:log10, color="blue")
plot!(plot1,1:iter,matrix_data_multi[:,1], yscale=:log10, color="magenta")
plot!(plot1,1:iter, matrix_data_hybrid, yscale=:log10, color="green")


# iter = 100
# plot1=plot(1:iter, matrix_data_single[1:iter,:],  yscale=:log10, color="red")
# plot!(plot1,1:iter, matrix_data_multi[1:iter,:],  yscale=:log10, color="blue")
# plot!(plot1,1:iter,matrix_data_multi[1:iter,1], yscale=:log10, color="magenta")


hist1=histogram(out_single_eps.population,bins=100,color="red")
histogram!(hist1,out_hybrid.population,bins=100,color="green")
histogram!(hist1,out_multi_eps.population,bins=100,color="blue")
histogram!(hist1,samples,bins=100,color="yellow")
