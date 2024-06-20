import Pkg
# Activate environment. 
Pkg.activate("SABC") 

#= Current environment is visible in Pkg REPL (type']' to activate Pkg REPL)
In Pkg REPL (activated with ']') do:
'add ...path to.../SimulatedAnnealingABC.jl' 
to have the local package installed
OR (after loading Revise)
'dev ...path to.../SimulatedAnnealingABC.jl' to develop package =#

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
using Dates
using LinearAlgebra

include("./AffineInvMCMC.jl")
using .AffineInvMCMC

println(" ")
println("---- ----------------------------- ----")
println("---- NLAR1 - Infer alpha and sigma ----")
println("---- ----------------------------- ----")

"""
-----------------------------------------------------------------
--- NLAR1 model:
--- Data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""

# --- Model ---
function nlf(x::Float64)
	return x^2*(1-x)
end

function nlar1(α, σ; x0 = 0.25, l = 200)
	x = [x0] 
	for ix in 2:l
		push!(x, α*nlf(last(x)) + σ*randn())
	end
	return x
end

# --- Data from Albert et al., SciPost Phys.Core 2022 ---
true_alpha = 5.3
true_sigma = 0.015

y_obs = vec(readdlm("/Users/ulzg/SABC/test_data/dataset_c53_s0015_p0025_p200_values.dat"))
# 
display(plot(y_obs, title = "NLAR1 data"))  # display it (if you want)
true_posterior = collect(readdlm("/Users/ulzg/SABC/test_data/truePosterior_c53_s0015_p0025_p200.dat")')

# --- Prior ---
a_min = 4.2
a_max = 5.8
s_min = 0.005
s_max = 0.025
prior = product_distribution(Uniform(a_min, a_max), Uniform(s_min, s_max))

"""
-----------------------------------------------------------------
--- Infer α and σ for NLAR1 model
--- Statistics: MLEs for α and σ, order parameter
--- Multi vs single epsilon 
--- Metric: Euclidean
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats ---------- ----")

# --- Summary stats: definition ---
function αhat(x::Vector{Float64})
	num = sum(x[2:end].* nlf.(x[1:end-1]))
	den = sum((nlf.(x)).^2)
	return num/den
end

function σhat(x::Vector{Float64})
	num = sum( (x[2:end] .- αhat(x) .* nlf.(x[1:end-1])).^2 )
	den = length(x)
	return num/den            # gives larger posterior (in sigma dimension) with single eps, needs rescaling
	# return sqrt(num/den)    # both single and multi-eps give true post, no need for rescaling 
end

function order_par(x::Vector{Float64})
	num = sum((nlf.(x)).^2)
	den = length(x)
	return num/den
end

function sum_stats(data)
	stat1 = αhat(data)
	stat2 = σhat(data) 
	stat3 = order_par(data)
	return [stat1, stat2, stat3]
end

ss_obs = sum_stats(y_obs)

# --- Model + distance functions ---
function model(θ)
	α, σ = θ
	y = nlar1(α, σ)
	return sum_stats(y)
end

function f_dist_3stats(θ)
    # Data-generating model
	ss = model(θ)
    # Distance
    rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
    return rho
end

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

# --- Run ---
out_singeps = sabc(f_dist_3stats, prior; n_particles = np, v = 1.0, n_simulation = ns)
display(out_singeps)

out_singeps_2 = update_population!(out_singeps, f_dist_3stats, prior; v = 1.0, n_simulation = ns)
display(out_singeps_2)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)


# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 3 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(5.2, 5.4, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="α - single eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 3 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="σ - single eps")
display(P_hist_s)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), title = "NLAR1 - 3 stats",
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :green, label="single eps")
display(P_scatter_1)

# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 3 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="NLAR1 - u - 3 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="NLAR1 - rho - 3 stats", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - α", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - σ", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[3,1:end], xaxis=:log, yaxis=:log, label="single eps - order par", 
		linecolor = :yellow, linewidth=3, thickness_scaling = 1),
display(P_r)


"""
-----------------------------------------------------------------
--- Infer α and σ for NLAR1 model
--- Statistics: MLEs for α and σ
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 2 stats ---------- ----")

# --- Summary stats: definition ---
function αhat(x::Vector{Float64})
	num = sum(x[2:end].* nlf.(x[1:end-1]))
	den = sum((nlf.(x)).^2)
	return num/den
end

function σhat(x::Vector{Float64})
	num = sum( (x[2:end] .- αhat(x) .* nlf.(x[1:end-1])).^2 )
	den = length(x)
	return num/den
end

function order_par(x::Vector{Float64})
	num = sum((nlf.(x)).^2)
	den = length(x)
	return num/den
end

function sum_stats(data)
	stat1 = αhat(data)
	stat2 = σhat(data) 
	return [stat1, stat2]
end

ss_obs = sum_stats(y_obs)

# --- Model + distance functions ---
function model(θ)
	α, σ = θ
	y = nlar1(α, σ)
	return sum_stats(y)
end

function f_dist_2stats(θ)
    # Data-generating model
	ss = model(θ)
    # Distance
    rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
    return rho
end

np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

# --- Run ---
out_singeps = sabc(f_dist_3stats, prior; n_particles = np, v = 1.0, n_simulation = ns)
display(out_singeps)

out_singeps_2 = update_population!(out_singeps, f_dist_3stats, prior; v = 1.0, n_simulation = ns)
display(out_singeps_2)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 2 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(5.2, 5.4, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="α - single eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 2 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :green2, fillalpha=0.5, label="σ - single eps")
display(P_hist_s)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), title = "NLAR1 - 2 stats",
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :green, label="single eps")
display(P_scatter_1)

# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="NLAR1 - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="NLAR1 - rho - 3 stats", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - α", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - σ", 
		linecolor = :green4, linewidth=3, thickness_scaling = 1)
display(P_r);

