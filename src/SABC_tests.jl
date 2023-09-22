import Pkg
# Activate environment. 
Pkg.activate("SABC") 

#= Current environment is visible in Pkg REPL (type']' to activate Pkg REPL)
In Pkg REPL (activated with ']') do:
'add /Users/ulzg/SABC/SimulatedAnnealingABC.jl' 
to have the local package installed
OR (after loading Revise)
'dev /Users/ulzg/SABC/SimulatedAnnealingABC.jl' to develop package =#

#= To add dependencies:
pkg> activate /Users/ulzg/SABC/SimulatedAnnealingABC.jl
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


"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std
--- Multi vs single epsilon 
--- Metric: Euclidean
-----------------------------------------------------------------
"""
# --- Data ---
yobs = rand(Normal(5, 10), 1000)  # generate data
# display(histogram(yobs, bins=20, title = "The data"))  # display it (if you want)
# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
ss_obs = [s1obs, s2obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),1000)
        # Summary stats
        s1 = mean(y); s2 = std(y)
		ss = [s1, s2]
        # Distance
        rho = euclidean(ss, ss_obs)
        return rho
end

function f_dist_euclidean_multeps(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),1000)
	# Summary stats
	s1 = mean(y); s2 = std(y)
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Prior ---
prior = product_distribution(Uniform(-10,20), Uniform(0., 30))

nsim = 300_000
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, n_simulation = nsim)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, n_simulation = nsim)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "mean - 2 stats")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "std - 2 stats")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Plot epsilons ---
P_eps = plot(title="Epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", linewidth=3, thickness_scaling = 1)
display(P_eps)


"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, median
--- Multi vs single epsilon 
--- Metric: Euclidean
-----------------------------------------------------------------
"""
# --- Data ---
yobs = rand(Normal(5, 10), 1000)  # generate data
# display(histogram(yobs, bins=20, title = "The data"))  # display it (if you want)
# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
s3obs = median(yobs)
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_withmedian(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),1000)
        # Summary stats
        s1 = mean(y); s2 = std(y); s3 = median(y)
		ss = [s1, s2, s3]
        # Distance
        rho = euclidean(ss, ss_obs)
        return rho
end

function f_dist_euclidean_multeps_withmedian(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),1000)
	# Summary stats
	s1 = mean(y); s2 = std(y); s3 = median(y)
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Prior ---
prior = product_distribution(Uniform(-10,20), Uniform(0., 30))

nsim = 300_000
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_withmedian, prior; n_particles = 1000, n_simulation = nsim)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_withmedian, prior; n_particles = 1000, n_simulation = nsim)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "mean - 3 stats with median")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "std - 3 stats with median")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Plot epsilons ---
P_eps = plot(title="Epsilon - 3 stats with median", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - median", linewidth=3, thickness_scaling = 1)
display(P_eps)


#= 
"""
**** Multivariate Normal (2d) ****
"""

mus = [3,5] # Mean
σ = [1, 2]  # Std
ρ = -0.25   # Correlation, to construct covariance matrix

Σ = [ [σ[1] ρ*σ[1]*σ[2]]; [ρ*σ[1]*σ[2] σ[2]] ]

mvn = MvNormal(mus, Σ)

# To plot
function f(x,y)
	pdf(mvn,[x,y])
end
x=range(start = 0, stop=10, step=0.1)
y=range(start = 0, stop=10, step=0.1)
display(surface(x,y,f))

yobs = rand(mvn, 1000)
ssobs = mean(yobs, dims=2)

function f_dist_single(θ)
	# Data-generating model
	y = rand(MvNormal([θ[1],θ[2]], Σ), 1000)
	# Summary stats
	ss = mean(y, dims=2)
	# Distance
	rho = sqrt(sum((ssobs .- ss) .^ 2) / 2)
	return rho
end

function f_dist_multi(θ)
	# Data-generating model
	y = rand(MvNormal([θ[1],θ[2]], Σ), 1000)
	# Summary stats
	ss = mean(y, dims=2)
	# Distance
	rho = [euclidean(ss[ix], ssobs[ix]) for ix in 1:2]
	# rho = sqrt(sum((ssobs .- ss) .^ 2) / 2)
	return rho
end

prior = product_distribution(Uniform(0,10), Uniform(0, 10))

res_single = sabc(f_dist_single, prior; n_particles = 1000, n_simulation = 100000)

pop_single = hcat(res_single.population...)
eps_single = hcat(res_single.state.ϵ_history...)
display(histogram(pop_single[1,:], bins=range(0, 10, length=21)))
display(histogram(pop_single[2,:], bins=range(0, 10, length=31))) 

P_single=plot(title="Distance threshold")
for ix in eachrow(eps_single)
	plot!(P_single, ix[1:end], xaxis=:log, yaxis=:log)
end
display(P_single)

writedlm("/Users/ulzg/SABC/output/post_population_bivnormtest_single.csv", pop_single, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_bivnormtest_single.csv", eps_single, ',')

res_multi = sabc(f_dist_multi, prior; n_particles = 1000, n_simulation = 100000)

pop_multi = hcat(res_multi.population...)
eps_multi = hcat(res_multi.state.ϵ_history...)
display(histogram(pop_multi[1,:], bins=range(0, 10, length=21)))
display(histogram(pop_multi[2,:], bins=range(0, 10, length=31))) 

P_multi=plot(title="Distance threshold")
for ix in eachrow(eps_multi)
	plot!(P_multi, ix[1:end], xaxis=:log, yaxis=:log)
end
display(P_multi)

writedlm("/Users/ulzg/SABC/output/post_population_bivnormtest_multi.csv", pop_multi, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_bivnormtest_multi.csv", eps_multi, ',')

 =#