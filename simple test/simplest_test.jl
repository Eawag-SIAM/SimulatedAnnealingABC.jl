import Pkg
# Activate environment. 
Pkg.activate("SABC") 
                   
using Random
using Distributions
using Statistics
using SimulatedAnnealingABC
using Plots
using BenchmarkTools
using CSV 
using DataFrames
using FFTW
using Distances
using DifferentialEquations
using StochasticDelayDiffEq
using SpecialFunctions
using DelimitedFiles
using Dates
using LinearAlgebra

include("./AffineInvMCMC.jl")
using .AffineInvMCMC


"""
-----------------------------------------------------------------
--- Generate data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""
# --- Data ---
Random.seed!(1111)
# Random.seed!()
true_μ = 3
true_σ = 15
num_samples = 100

y_obs = rand(Normal(true_μ, true_σ), num_samples)  # generate data
# display(histogram(yobs, bins=20, title = "The data"))  # display it (if you want)

# --- Prior ---
μ_min = -10; μ_max = 20    # parameter 1: mean
σ_min = 0; σ_max = 25      # parameter 2: std
prior = product_distribution(Uniform(μ_min, μ_max), Uniform(σ_min, σ_max))

# --- True posterior ---
llhood = theta -> begin
	m, s  = theta;
	return -num_samples*log(s) - sum((y_obs.-m).^2)/(2*s^2)
end

lprior = theta -> begin
	m, s = theta;
	if (μ_min <= m <= μ_max) && (σ_min <= s <= σ_max)
		return 0.0
	else
		return -Inf
	end
end

lprob = theta -> begin
	m,s = theta;
	lp = lprior(theta)
	if isinf(lp) 
		return -Inf
	else
		return lp + llhood(theta)
	end
end

numdims = 2
numwalkers = 10
thinning = 10
numsamples_perwalker = 1000
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(μ_min, μ_max), numwalkers);  # mu
theta0[2, :] = rand(rng, Uniform(σ_min, σ_max), numwalkers);  # sigma

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

########################################################################################
### Plot True Posterior
########################################################################################
P_scatter_1 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), title = "1d Normal - True posterior",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true posterior")
P_scatter_2 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), xlabel = "std")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true posterior")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
########################################################################################

########################################################################################
### Reset RNG seed for the inference
### Use Random.seed!() for random inference outputs  
Random.seed!(1807)
# Random.seed!()
########################################################################################

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal
--- Statistics: mean, std
-----------------------------------------------------------------
"""
function sum_stats(data)
	stat1 = mean(data)
	stat2 = std(data) 
	return [stat1, stat2]
end

n_stats = size(sum_stats(y_obs), 1)

# --- Summary stats ---
ss_obs = sum_stats(y_obs)

# --- Model + distance functions ---
function model(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return sum_stats(y)
end

function f_dist(θ)
	# Data-generating model
	ss = model(θ)
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

np = 1000       # number of particles
ns = 5_000_000  # number of particle updates


# --- Run for Differential Evolution ---
out_dif = sabc(f_dist, prior; n_particles = np, v = 1.0, n_simulation = ns, 
				algorithm = :single_eps, proposal = DifferentialEvolution(n_para = 2))
# --- Run for Stretch Move ---
out_str = sabc(f_dist, prior; n_particles = np, v = 1.0, n_simulation = ns,
				algorithm = :single_eps, proposal = StretchMove())
# --- Run for Random Walk ---
out_rnd = sabc(f_dist, prior; n_particles = np, v = 1.0, n_simulation = ns,
				algorithm = :single_eps, proposal = RandomWalk(n_para = 2))


pop_dif = hcat(out_dif.population...)
eps_dif = hcat(out_dif.state.ϵ_history...)
rho_dif = hcat(out_dif.state.ρ_history...)
u_dif = hcat(out_dif.state.u_history...)

pop_str = hcat(out_str.population...)
eps_str = hcat(out_str.state.ϵ_history...)
rho_str = hcat(out_str.state.ρ_history...)
u_str = hcat(out_str.state.u_history...)

pop_rnd = hcat(out_rnd.population...)
eps_rnd = hcat(out_rnd.state.ϵ_history...)
rho_rnd = hcat(out_rnd.state.ρ_history...)
u_rnd = hcat(out_rnd.state.u_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 2 stats")
histogram!(P_hist_mu, pop_dif[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="mean - DiffEv")
histogram!(P_hist_mu, pop_str[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - Stretch")
histogram!(P_hist_mu, pop_rnd[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :blue, fillalpha=0.5, label="mean - RndWalk")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 2 stats")
histogram!(P_hist_sd, pop_dif[2,:], bins=range(0, 30, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="std - DiffEv")
histogram!(P_hist_sd, pop_str[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - Stretch")
histogram!(P_hist_sd, pop_rnd[2,:], bins=range(0, 30, length=31), 
			fillcolor = :blue, fillalpha=0.5, label="std - RndWalk")
display(P_hist_sd)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (σ_min, σ_max), ylims = (μ_min,μ_max), title = "1d Normal - 2 stats",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label=" true post ")
scatter!(P_scatter_1, pop_dif[2,:], pop_dif[1,:], markercolor = :green2, label=" DiffEv ")
P_scatter_2 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), xlabel = "std")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label=" true post ")
scatter!(P_scatter_2, pop_str[2,:], pop_str[1,:], markercolor = :coral, label=" Stretch ")
P_scatter_3 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), xlabel = "std")
scatter!(P_scatter_3, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label=" true post ")
scatter!(P_scatter_3, pop_rnd[2,:], pop_rnd[1,:], markercolor = :blue, label=" RndWalk ")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1, P_scatter_2, P_scatter_3, layout = (1, 3)))

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_dif[1,1:end], xaxis=:log, yaxis=:log, label="DiffEv", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_str[1,1:end], xaxis=:log, yaxis=:log, label="Stretch", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_rnd[1,1:end], xaxis=:log, yaxis=:log, label="RndWalk", 
		linecolor = :blue, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="1d Normal - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_dif[1,1:end], xaxis=:log, yaxis=:log, label="DiffEv - mean", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_dif[2,1:end], xaxis=:log, yaxis=:log, label="DiffEv - std", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_str[1,1:end], xaxis=:log, yaxis=:log, label="Stretch - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_str[2,1:end], xaxis=:log, yaxis=:log, label="Stretch - std", 
		linecolor = :red, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_rnd[1,1:end], xaxis=:log, yaxis=:log, label="RndWalk - mean", 
		linecolor = :blue, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_rnd[2,1:end], xaxis=:log, yaxis=:log, label="RndWalk - std", 
		linecolor = :darkblue, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="1d Normal - rho - 2 stats", legend = :bottomleft)
plot!(P_r, rho_dif[1,1:end], xaxis=:log, yaxis=:log, label="DiffEv - mean", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_dif[2,1:end], xaxis=:log, yaxis=:log, label="DiffEv - std", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_str[1,1:end], xaxis=:log, yaxis=:log, label="Stretch - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_str[2,1:end], xaxis=:log, yaxis=:log, label="Stretch - std", 
		linecolor = :red, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_rnd[1,1:end], xaxis=:log, yaxis=:log, label="RndWalk - mean", 
		linecolor = :blue, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_rnd[2,1:end], xaxis=:log, yaxis=:log, label="RndWalk - std", 
		linecolor = :darkblue, linewidth=3, thickness_scaling = 1)
display(P_r)
##################################################################

##################################################################
### Kullback - Leibler divergence in 2D
#################################################################
using KernelDensity
using QuadGK
using LinearAlgebra

# 2D samples (n x 2 arrays)
true_samples = hcat(flatchain[2,:], flatchain[1,:])
sampled1_samples = hcat(pop_dif[2,:], pop_dif[1,:])
sampled2_samples = hcat(pop_str[2,:], pop_str[1,:])
sampled3_samples = hcat(pop_rnd[2,:], pop_rnd[1,:])

# Perform 2D KDE
kde_q = kde(true_samples)   # transpose is needed for 2D samples
kde_p1 = kde(sampled1_samples)
kde_p2 = kde(sampled2_samples)
kde_p3 = kde(sampled3_samples)

# Determine ranges for x and y based on all samples
x_min = minimum([minimum(true_samples[:, 1]), minimum(sampled1_samples[:, 1]),
			minimum(sampled2_samples[:, 1]), minimum(sampled3_samples[:, 1])])
x_max = maximum([maximum(true_samples[:, 1]), maximum(sampled1_samples[:, 1]),
			maximum(sampled2_samples[:, 1]), minimum(sampled3_samples[:, 1])])

y_min = minimum([minimum(true_samples[:, 2]), minimum(sampled1_samples[:, 2]),
			minimum(sampled2_samples[:, 2]), minimum(sampled3_samples[:, 2])])
y_max = maximum([maximum(true_samples[:, 2]), maximum(sampled1_samples[:, 2]),
			maximum(sampled2_samples[:, 2]), maximum(sampled3_samples[:, 2])])

function monte_carlo_kl_divergence(kde_p, kde_q, x_min, x_max, y_min, y_max; num_samples=1000)
    total = 0.0
    for _ in 1:num_samples
        x = rand() * (x_max - x_min) + x_min
        y = rand() * (y_max - y_min) + y_min
        p_xy = pdf(kde_p, x, y)
        q_xy = pdf(kde_q, x, y)
        
        # Avoid division by zero
        if p_xy > 1e-7 && q_xy > 1e-7
            total += q_xy * log(q_xy / p_xy)
        end
    end
    
    area = (x_max - x_min) * (y_max - y_min)
    return total * area / num_samples
end

# Calculate KL divergence for each pair
asa =10
kl_divergence_p1_q = 0
kl_divergence_p2_q = 0
kl_divergence_p3_q = 0
for ix = 1:asa
	println("Integration ", ix, " of ", asa)
	kl_divergence_p1_q += monte_carlo_kl_divergence(kde_p1, kde_q, x_min, x_max, y_min, y_max; num_samples=2000)
	kl_divergence_p2_q += monte_carlo_kl_divergence(kde_p2, kde_q, x_min, x_max, y_min, y_max; num_samples=2000)
	kl_divergence_p3_q += monte_carlo_kl_divergence(kde_p3, kde_q, x_min, x_max, y_min, y_max; num_samples=2000)
end
println("KL Divergence Q || P1 (2D): ", kl_divergence_p1_q / asa)
println("KL Divergence Q || P2 (2D): ", kl_divergence_p2_q / asa)
println("KL Divergence Q || P3 (2D): ", kl_divergence_p3_q / asa)
##################################################################
### End of KL divergence
#################################################################