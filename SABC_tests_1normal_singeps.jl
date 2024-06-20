import Pkg
# Activate environment. 
Pkg.activate("SABC") 

#= Current environment is visible in Pkg REPL (type ']' to activate Pkg REPL)
In Pkg REPL (activated with ']') do:
'add ...path to.../SimulatedAnnealingABC.jl' 
to have the local package installed
OR (after loading Revise)
'dev ...path to.../SimulatedAnnealingABC.jl' to develop package =#

#= To add dependencies:
pkg> activate ...path to.../SimulatedAnnealingABC.jl
pkg> add PkgName =#
                   
using Revise
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

println(" ")
println("---- ------------------------------ ----")
println("---- 1d Normal - Infer mean and std ----")
println("---- ------------------------------ ----")

"""
-----------------------------------------------------------------
--- Generate normally distributed data
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
### Wanna plot the true posterior ?
### Run this!
########################################################################################
P_scatter = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), title = "1d Normal - True posterior",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true posterior")
display(P_scatter)
########################################################################################

########################################################################################
### Reset RNG seed for the inference
Random.seed!(1807)
# Random.seed!()
########################################################################################

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 2 stats ---------- ----")

function sum_stats(data)
	stat1 = mean(data)
	stat2 = std(data) 
	return [stat1, stat2]
end

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
ns = 1_000_000  # number of particle updates

# --- Run ---
out_singeps = sabc(f_dist, prior; n_particles = np, v = 1.0, n_simulation = ns)
display(out_singeps)

out_singeps_2 = update_population!(out_singeps, f_dist, prior; v = 1.0, n_simulation = ns)
display(out_singeps_2)

pop_singeps = hcat(out_singeps_2.population...)
eps_singeps = hcat(out_singeps_2.state.ϵ_history...)
rho_singeps = hcat(out_singeps_2.state.ρ_history...)
u_singeps = hcat(out_singeps_2.state.u_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 2 stats")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="mean - single eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 2 stats")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="std - single eps")
display(P_hist_sd)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (σ_min, σ_max), ylims = (μ_min,μ_max), title = "1d Normal - 2 stats",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label=" true post ")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :green, label=" SABC ")
display(P_scatter_1)

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="1d Normal - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="1d Normal - rho - 2 stats", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
display(P_r)
##################################################################

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, median
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats (median) ---------- ----")

function sum_stats_withmedian(data)
	stat1 = mean(data)
	stat2 = std(data) 
	stat3 = median(data)
	return [stat1, stat2, stat3]
end

# --- Summary stats ---
ss_obs = sum_stats_withmedian(y_obs)

# --- Model + distance functions ---
function model_withmedian(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return sum_stats_withmedian(y)
end

function f_dist_withmedian(θ)
	# Data-generating model
	ss = model_withmedian(θ)
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

# --- Run ---
out_singeps = sabc(f_dist_withmedian, prior; n_particles = np, v = 1.0, n_simulation = ns)
display(out_singeps)

out_singeps_2 = update_population!(out_singeps, f_dist_withmedian, prior; v = 1.0, n_simulation = ns)
display(out_singeps_2)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with median")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="mean - single eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with median")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="std - single eps")
display(P_hist_sd)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), title = "1d Normal - 3 stats with median",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :green, label="single eps")
display(P_scatter_1)

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with median", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="1d Normal - u - 3 stats with median", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="1d Normal - rho - 3 stats with median", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[3,1:end], xaxis=:log, yaxis=:log, label="single eps - median", 
		linecolor = :yellow, linewidth=3, thickness_scaling = 1),
display(P_r)

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, some noise
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats (noise) ---------- ----")

function sum_stats_withnoise(data)
	stat1 = mean(data)
	stat2 = std(data) 
	stat3 = randn()
	return [stat1, stat2, stat3]
end

# --- Summary stats ---
ss_obs = sum_stats_withnoise(y_obs)

# --- Model + distance functions ---
function model_withnoise(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return sum_stats_withnoise(y)
end

function f_dist_withnoise(θ)
	# Data-generating model
	ss = model_withnoise(θ)
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

# --- Run ---
out_singeps = sabc(f_dist_withnoise, prior; n_particles = np, v = 1.0, n_simulation = ns)
display(out_singeps)

out_singeps_2 = update_population!(out_singeps, f_dist_withnoise, prior; v = 1.0, n_simulation = ns)
display(out_singeps_2)


# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with noise")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="mean - single eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with noise")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="std - single eps")
display(P_hist_sd)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), title = "1d Normal - 3 stats with noise",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :green, label="SABC")
display(P_scatter_1)

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with noise", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="1d Normal - u - 3 stats with noise", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="1d Normal - rho - 3 stats with noise", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[3,1:end], xaxis=:log, yaxis=:log, label="single eps - noise", 
		linecolor = :yellow, linewidth=3, thickness_scaling = 1),
display(P_r);

