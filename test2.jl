using Random
using Distributions
using Distances
using SimulatedAnnealingABC
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

function f_dist(param)
    # Draw from the distribution
    y = sample_mixture(Normal(param,1),Normal(-param,sigma),mixing_coeff,2)
    s1=y[1]+y[2]
    s2=y[1]^2+y[2]^2

    y_others = rand(Normal(0, 1), 9)
    s_others = abs.(y_others)

    return vcat([abs(s1-sobs1),abs(s2-sobs2)],s_others)
end

function posterior_sample(y1, y2, α, σ, n_samples=1000)
    # Mixture component means and variances
    means = [
        (y1 + y2) / 2,
        -(y1 + y2) / 2
    ]
    variances = [
        1 / 2,
        σ^2/ 2
    ]
    stddevs = sqrt.(variances)

    # Compute unnormalized weights
    component_likelihoods = [
        α*α* exp(((y1+y2)/2)^2-(y1^2+y2^2)/2 ),
        (1 - α) * (1 - α) * exp( ( (y1+y2)/(2*σ) )^2-(y1^2+y2^2)/(2*σ^2) )/σ
    ]
    weights = component_likelihoods / sum(component_likelihoods)

    # Define the mixture distribution
    components = [Normal(mean, std) for (mean, std) in zip(means, stddevs)]
    mix_dist = MixtureModel(components, weights)

    # Sample from the posterior distribution
    rand(mix_dist, n_samples)
end

Random.seed!(1111)					  

# Set parameters
mixing_coeff = 0.3
sigma = 0.3
prior = Uniform(-10,10)

np = 1000       # number of particles
ns = 3_000_000  # number of particle updates

# synthetic data and summary stats:
# yobs = sample_mixture(Normal(param_true,1),Normal(-param_true,sigma),mixing_coeff,2)

yobs = (5,5)
sobs1 = yobs[1]+yobs[2]
sobs2 = yobs[1]^2+yobs[2]^2

# true (approx) posterior
samples = posterior_sample(yobs[1], yobs[2], mixing_coeff, sigma, np)
histogram(samples,bins=100)

# --- TYPE 1 -> single-ϵ ---
out_single_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0, type = 1)
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10

out_single_eps = update_population!(out_single_eps, f_dist, prior; n_simulation = ns, v = 1.0, type = 1)
data = out_single_eps.state.ρ_history
matrix_data_single = hcat(data...)'  # Transpose after concatenation to get 1000x10

# --- TYPE 2 -> multi-ϵ ---
out_multi_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 0.01, type = 2)
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
plot!(plot1,1:iter, matrix_data_hybrid, yscale=:log10, color="green")


hist1=histogram(out_single_eps.population,bins=100,color="red")
histogram!(hist1,out_hybrid.population,bins=100,color="green")
histogram!(hist1,out_multi_eps.population,bins=100,color="blue")
histogram!(hist1,samples,bins=100,color="yellow")
