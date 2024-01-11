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
Random.seed!(1822)

true_mean = 3
true_sigma = 15
num_samples = 100

yobs = rand(Normal(true_mean, true_sigma), num_samples)  # generate data
# display(histogram(yobs, bins=20, title = "The data"))  # display it (if you want)

# --- Prior ---
s1_min = -10
s1_max = 20
s2_min = 0
s2_max = 25
prior = product_distribution(Uniform(s1_min, s1_max), Uniform(s2_min, s2_max))

# --- True posterior ---
llhood = theta -> begin
	m, s  = theta;
	return -length(yobs)*log(s) - sum((yobs.-m).^2)/(2*s^2)
end

lprior = theta -> begin
	m, s = theta;
	if (s1_min <= m <= s1_max) && (s2_min <= s <= s2_max)
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

# We want num_samples posterior samples
numdims = 2
numwalkers = 10
thinning = 10
numsamples_perwalker = num_samples
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(s1_min, s1_max), numwalkers);  # mu
theta0[2, :] = rand(rng, Uniform(s2_min, s2_max), numwalkers);  # sigma

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

########################################################################################
### Wanna plot the true posterior ?
### Run this!
########################################################################################
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), title = "1d Normal - True posterior",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
########################################################################################


nsim = 2_000_000  # total number of particle updates 
# N.B.: for NLAR1, 'nsim' is re-defined below

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 2 stats ---------- ----")

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
ss_obs = [s1obs, s2obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]), num_samples)
        # Summary stats
        s1 = mean(y); s2 = std(y)
		ss = [s1, s2]
        # Distance
        rho = euclidean(ss, ss_obs)
        return rho
end

function f_dist_euclidean_multeps(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]), num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y)
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

##################################################################

##################################################################
### Wanna run multiple trajectories ?
### Run this!
#################################################################
#= 
num_traj = 200
# --- Run for single distance ---
out_singeps = Array{Any}(undef, num_traj)
for id in 1:num_traj
	out_singeps[id] = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, n_simulation = nsim, v = 1.2)
end
# --- Run for multiple distances ---
out_multeps = Array{Any}(undef, num_traj)
for id in 1:num_traj
	out_multeps[id] = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, n_simulation = nsim, v = 1.2)
end

# --- Extract populations and epsilons ---
pop_singeps = Array{Any}(undef, num_traj)
eps_singeps = Array{Any}(undef, num_traj)
rho_singeps = Array{Any}(undef, num_traj)
u_singeps = Array{Any}(undef, num_traj)
for ix in 1:num_traj
	pop_singeps[ix] = hcat(out_singeps[ix].population...)
	eps_singeps[ix] = hcat(out_singeps[ix].state.ϵ_history...)
	rho_singeps[ix] = hcat(out_singeps[ix].state.ρ_history...)
	u_singeps[ix] = hcat(out_singeps[ix].state.u_history...)
end

pop_multeps = Array{Any}(undef, num_traj)
eps_multeps = Array{Any}(undef, num_traj)
rho_multeps = Array{Any}(undef, num_traj)
u_multeps = Array{Any}(undef, num_traj)
for ix in 1:num_traj
	pop_multeps[ix] = hcat(out_multeps[ix].population...)
	eps_multeps[ix] = hcat(out_multeps[ix].state.ϵ_history...)
	rho_multeps[ix] = hcat(out_multeps[ix].state.ρ_history...)
	u_multeps[ix] = hcat(out_multeps[ix].state.u_history...)
end

fname_singeps = "test_normal_2stats_singeps_100samples_seed1113"
fname_multeps = "test_normal_2stats_multeps_100samples_seed1113"

for ix in 1:num_traj
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/post_population_" * fname_singeps * "_" * string(ix) * ".csv", pop_singeps[ix], ',') 
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/epsilon_history_" * fname_singeps * "_" * string(ix) * ".csv", eps_singeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/rho_history_" * fname_singeps * "_" * string(ix) * ".csv", rho_singeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/u_history_" * fname_singeps * "_" * string(ix) * ".csv", u_singeps[ix], ',')

	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/post_population_" * fname_multeps * "_" * string(ix) * ".csv", pop_multeps[ix], ',') 
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/epsilon_history_" * fname_multeps * "_" * string(ix) * ".csv", eps_multeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/rho_history_" * fname_multeps * "_" * string(ix) * ".csv", rho_multeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/u_history_" * fname_multeps * "_" * string(ix) * ".csv", u_multeps[ix], ',')
end
 =#
##################################################################

##################################################################
### Single trajectory?
### Run this!
#################################################################
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, n_simulation = nsim, v = 1.2)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, n_simulation = nsim, v = 84)
display(out_multeps)

pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

##################################################################
### Wanna save outputs ?
### Run this!
#################################################################
#= 
fname_singeps = "test_normal_2stats_singeps_100samples_seed1822"
fname_multeps = "test_normal_2stats_multeps_100samples_seed1822"

writedlm("/Users/ulzg/SABC/output/post_population_" * fname_singeps * ".csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_" * fname_singeps * ".csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_" * fname_singeps * ".csv", rho_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_" * fname_singeps * ".csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_" * fname_multeps * ".csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_" * fname_multeps * ".csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_" * fname_multeps * ".csv", rho_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_" * fname_multeps * ".csv", u_multeps, ',')
 =#

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 2 stats")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 2 stats")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), title = "1d Normal - 2 stats",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_eps)

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

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
s3obs = median(yobs)
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_withmedian(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),num_samples)
        # Summary stats
        s1 = mean(y); s2 = std(y); s3 = median(y)
		ss = [s1, s2, s3]
        # Distance
        rho = euclidean(ss, ss_obs)
        return rho
end

function f_dist_euclidean_multeps_withmedian(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y); s3 = median(y)
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_withmedian, prior; n_particles = 1000, n_simulation = nsim,  v = 1.2)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_withmedian, prior; n_particles = 1000, n_simulation = nsim,  v = 84)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with median")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with median")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), 
					xlabel = "std", ylabel = "mean", title = "1d Normal - 3 stats with median")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with median", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - median",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, some noise
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats (noise) ---------- ----")

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
s3obs = randn()
# s3obs = rand(Normal(0, 5))
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_withnoise(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),num_samples)
        # Summary stats
        s1 = mean(y); s2 = std(y); s3 = randn() # rand(Normal(0, 5)) 
		ss = [s1, s2, s3]
        # Distance
        rho = euclidean(ss, ss_obs)
        return rho
end

function f_dist_euclidean_multeps_withnoise(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y); s3 = randn() # rand(Normal(0, 5)) 
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_withnoise, prior; n_particles = 1000, n_simulation = nsim,  v = 1.2)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_withnoise, prior; n_particles = 1000, n_simulation = nsim,  v = 84)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with noise stat")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with noise stat")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "1d Normal - 3 stats with noise stat", xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), 
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with noise stat", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - noise stat",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)


println(" ")
println("---- ----------------------------- ----")
println("---- NLAR1 - Infer alpha and sigma ----")
println("---- ----------------------------- ----")

sleep(0.5)

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

yobs = vec(readdlm("/Users/ulzg/SABC/SimulatedAnnealingABC.jl/src/test_data/dataset_c53_s0015_p0025_p200_values.dat"))
# 
display(plot(yobs, title = "NLAR1 data"))  # display it (if you want)
true_posterior = collect(readdlm("/Users/ulzg/SABC/SimulatedAnnealingABC.jl/src/test_data/truePosterior_c53_s0015_p0025_p200.dat")')

# --- Prior ---
a_min = 4.2
a_max = 5.8
s_min = 0.005
s_max = 0.025
prior = product_distribution(Uniform(a_min, a_max), Uniform(s_min, s_max))


nsim = 2_000_000  # total number of particle updates

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
	return num/den
end

function order_par(x::Vector{Float64})
	num = sum((nlf.(x)).^2)
	den = length(x)
	return num/den
end


# --- Summary stats: data ---
s1obs = αhat(yobs)
s2obs = σhat(yobs) 
s3obs = order_par(yobs)
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_3stats(θ)
    α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y); s3 = order_par(y)
	ss = [s1, s2, s3]
    # Distance
    rho = euclidean(ss, ss_obs)
    return rho
end

function f_dist_euclidean_multeps_3stats(θ)
	α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y); s3 = order_par(y)
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_3stats, prior; n_particles = 1000, n_simulation = nsim,  v = 1.2)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_3stats, prior; n_particles = 1000, n_simulation = nsim,  v = 84)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 3 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="alpha - single eps")
histogram!(P_hist_a, pop_multeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="alpha - multi eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 3 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="sigma - single eps")
histogram!(P_hist_s, pop_multeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="sigma - multi eps")
display(P_hist_s)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "NLAR1 - 3 stats", xlims = (s_min,s_max), ylims = (a_min,a_max), 
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), xlabel = "σ")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 3 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - α", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - σ", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - order par",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)

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


# --- Summary stats: data ---
s1obs = αhat(yobs)
s2obs = σhat(yobs) 
ss_obs = [s1obs, s2obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_3stats(θ)
    α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y)
	ss = [s1, s2]
    # Distance
    rho = euclidean(ss, ss_obs)
    return rho
end

function f_dist_euclidean_multeps_3stats(θ)
	α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y);
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_3stats, prior; n_particles = 1000, n_simulation = nsim,  v = 1.2)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_3stats, prior; n_particles = 1000, n_simulation = nsim,  v = 84)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)

# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 2 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="alpha - single eps")
histogram!(P_hist_a, pop_multeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="alpha - multi eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 2 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="sigma - single eps")
histogram!(P_hist_s, pop_multeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="sigma - multi eps")
display(P_hist_s)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "NLAR1 - 2 stats", xlims = (s_min,s_max), ylims = (a_min,a_max), 
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), xlabel = "σ")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - α", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - σ", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
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