using Random
using Distributions
using Distances
using SimulatedAnnealingABC
# using Pkg
using Plots

# Function to sample from the mixture of two Gaussians
function sample_mixture(dist1, dist2, mixing_coeff, n_samples)
    samples = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        if rand() < mixing_coeff[1]
            samples[i] = rand(dist1)
        else
            samples[i] = rand(dist2)
        end
    end
    return samples
end

Random.seed!(1111)					  

prior = Uniform(-10,10)

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

mixing_coeff = 0.3
sigma = 0.1

param_true = 0.5
yobs = sample_mixture(Normal(param_true,1),Normal(-param_true,sigma),mixing_coeff,2)
sobs1 = yobs[1]+yobs[2]
sobs2 = yobs[1]-yobs[2]

#true_sample = sample_mixture(Normal(yobs,1),Normal(-yobs,sigma),mixing_coeff,np)
#histogram(true_sample,bins = 50, color="yellow")


function f_dist(param)
    # Draw from the distribution
    y = sample_mixture(Normal(param,1),Normal(-param,sigma),mixing_coeff,2)
    s1=y[1]+y[2]
    s2=y[1]-y[2]
    return [abs(s1-sobs1),abs(s2-sobs2)]
end

# --- TYPE 1 -> single-ϵ ---
out_single_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 1)
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10

# --- TYPE 2 -> multi-ϵ ---
out_multi_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 10.0, type = 2)
data = out_multi_eps.state.ρ_history
matrix_data_multi = hcat(data...)'  # Transpose after concatenation to get 1000x10

# --- TYPE 3 -> hybrid multi-u-single-ϵ ---
out_hybrid = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 3)
data = out_hybrid.state.ρ_history
matrix_data_hybrid = hcat(data...)'  # Transpose after concatenation to get 1000x10

# out_single_eps_2 = update_population!(out_single_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 1)
# out_multi_eps_2 = update_population!(out_multi_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 2)

iter = size(data)[1]

plot1=plot(1:iter, matrix_data_single,  yscale=:log10, color="red")
plot!(plot1,1:iter, matrix_data_multi,  yscale=:log10, color="blue")
plot!(plot1,1:iter, matrix_data_hybrid, yscale=:log10, color="green")


hist1=histogram(out_single_eps.population,bins=30,color="red")
histogram!(hist1,out_multi_eps.population,bins=30,color="blue")
histogram!(hist1,out_hybrid.population,bins=30,color="green")
histogram!(hist1,true_sample,bins=30,color="yellow")
