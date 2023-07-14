module SimulatedAnnealingABC


import Base.show
using LinearAlgebra
using Random

using ElasticArrays
using UnPack: @unpack
import ProgressMeter

import StatsBase
using StatsBase: mean, cov, sample, weights
using Interpolations: interpolate, extrapolate,
    LinearMonotonicInterpolation, SteffenMonotonicInterpolation, Flat
using Distributions: Distribution, pdf, MvNormal, Normal
import Roots


export sabc, update_population!


# -----------
# define types to hold results

"""
Holds states of algorithm
"""
mutable struct SABCstate
    ϵ::Vector{Float64}
    cdf_dist_prior::Vector                   # functions G in Albert et al.
    Σ_jump::Union{Matrix{Float64}, Float64}  # Float64 for 1d
    n_simulation::Int
    n_accept::Int
end

"""
Holds results

- `population`: vector of parameteer samples from the approximative posterior
- `u`: transformed distances
- `state`: state of algorithm
"""
struct SABCresult{T, S}
    population::Vector{T}
    u::Array{S}
    state::SABCstate
end

# Functions for pretty printing
function show(io::Base.IO, s::SABCresult)
    n_particles = length(s.population)
    mean_u = mean(s.u)

    println(io, "Approximate posterior sample with $n_particles particles:")
    println(io, "  - simulations used: $(s.state.n_simulation)")
    println(io, "  - average transformed distance: $mean_u")
    println(io, "  - ϵ: $(s.state.ϵ)")
    println(io, "The sample can be accessed with the field `population`.")
end



# -----------
# algorithm


"""
Solve for ϵ

See eq(31)
"""
 function update_epsilon(u, v)
     mean_u = mean(u)
     ϵ_new = mean_u <= eps() ? zero(mean_u) : Roots.find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - mean_u^2, (0, mean_u))
     ϵ_new
end


"""
Resample population

"""
function resample_population!(population, u, δ)
    n = length(population)
    u_means = mean(u, dims=1)
    w = exp.(-sum(u[:,i] .* δ ./ u_means[i] for i in 1:size(u, 2)))

    idx_resampled = sample(1:n, weights(w), n, replace=true)

    permute!(population, idx_resampled)
    permute!(u, idx_resampled)

    @info "Resampling. Effective sample size: $(round(1/sum(abs2, w ./ sum(w)), digits=2))"
end


"""
Estimate the coavaqriance for the jump distributions from an population
"""
function estimate_jump_covariance(population, β)
    β * cov(stack(population, dims=1)) + 1e-6*I
end


"""
Estimate the cdf of data `x`.
Returns a function.
"""
function build_cdf(x)
    StatsBase.ecdf(x)
end

"""
Estimate the empirical cdf of data `x`
smoothed by interpolation.

Returns a function.
"""
function build_cdf_smoothed(x)
    n = length(x)
    # note, we add a 0 and 1 probability points
    probs = [0; [(k - 0.5)/n for k in 1:n]; 1]
    a = 1.5
    values = [0; sort(x); maximum(x)*a]

    extrapolate(interpolate(values, probs,
                            LinearMonotonicInterpolation()
                            #SteffenMonotonicInterpolation()
                            ),
                Flat())

end



"""
Proposal for n-dimensions, n > 1
"""
proposal(θ, Σ::Matrix) = θ .+ rand(MvNormal(zeros(size(Σ,1)), Σ))


"""
Proposal for 1-dimensions
"""
proposal(θ, Σ::Float64) = θ + rand(Normal(0, Σ))


"""
# Initialisation step

## Arguments
See docs for `sabc`.

## Value

- population
- u
- state
"""
function initialization(f_dist, prior::Distribution, args...;
                        n_particles, n_simulation, eps_init,
                        v=1.2, β=0.8, kwargs...)

    ## ------------------------
    ## Initialize containers

    θ = rand(prior)
    ρ = f_dist(θ, args...; kwargs...)
    n_stats = length(ρ)

    population = Vector{typeof(θ)}(undef, n_particles)
    distances = Array{eltype(ρ)}(undef, n_particles, n_stats)
    distances_prior = ElasticArray{eltype(ρ)}(undef, n_stats, 0)


    ## ------------------
    ## Build prior sample

    iter = 0
    counter = 0 # Number of accepted particles in population
    progbar = ProgressMeter.Progress(n_particles, desc="Generating initial population...", dt=0.5)
    while counter < n_particles

        iter += 1
        if iter > n_simulation
            error("The initial population could not be generated. 'n_simulation' is to small!")
        end

        ## Generate new particle
        θ = rand(prior)
        ρ = f_dist(θ, args...; kwargs...)

        ## store distance
        append!(distances_prior, ρ)

        ## Accept with Prob = exp(-rho/eps.init) and store in population
        if rand() < exp(-sum(ρ ./ eps_init))
            counter += 1
            population[counter] = θ
            distances[counter,:] .= ρ
            ProgressMeter.next!(progbar)
        end
    end

    ## ----------------------------------
    ## Compute ϵ for every statistic

    ## empirical cdfs of ρ under the prior
    any(distances_prior .< 0) && error("Negative distances are not allowed!")
    cdf_dist_prior = map(build_cdf_smoothed, eachrow(distances_prior))

    ϵ = zeros(n_stats)
    u = similar(distances)
    for i in 1:n_stats
        u[:,i] .= cdf_dist_prior[i].(distances[:,i])
        ϵ[i] = update_epsilon(u[:,i], v)
    end

    Σ_jump = estimate_jump_covariance(population, β)

    # collect all parameters and states of the algorithm
    n_simulation = length(distances_prior)

    state = SABCstate(ϵ,
                      cdf_dist_prior,
                      Σ_jump,
                      n_simulation,
                      0)

    @info "Population with $n_particles particles initialised using $n_simulation simulation."

    return SABCresult(population, u, state)

end


"""
Updates particles and applies importance sampling if needed. Modifies `population_state`.

## Arguments

See `sabc`


"""
function update_population!(population_state::SABCresult, f_dist, prior, args...;
                            n_simulation,
                            v=0.3, β=1.5, δ=0.9,
                            resample=length(population_state.population),
                            kwargs...)

    @unpack population, u, state = population_state
    @unpack ϵ, n_accept, Σ_jump, cdf_dist_prior = state
    dim_par = length(first(population))
    n_particles = length(population)
    n_stats = length(ϵ)

    ## number of calls of `f_dist`
    n_updates = (n_simulation ÷ n_particles) * n_particles

    progbar = ProgressMeter.Progress(n_updates, desc="Updating...", dt=0.5)
    show_summary(state, u) = () -> [(:eps, state.ϵ), (:mean_transformed_distances, mean(u, dims=1))]

    for _ in 1:(n_simulation ÷ n_particles)

        ## -- update all particles (this can be multithreaded)
        for i in eachindex(population)

            # proposal
            θproposal = proposal(population[i], Σ_jump)

            # acceptance probability
            if pdf(prior, θproposal) > 0
                dists = f_dist(θproposal, args...; kwargs...)
                u_proposal = [cdf_dist_prior[k](dists[k]) for k in 1:n_stats]
                accept_prob = pdf(prior, θproposal) / pdf(prior, population[i]) *
                    exp(-sum((u_proposal .- u[i,:]) ./ ϵ))

            else
                accept_prob = 0.0
            end
            if rand() < accept_prob
                population[i] = θproposal
                u[i,:] .= u_proposal # transformed distances
                n_accept += 1
            end

        end

        ## -- update epsilon and jump distribution
        Σ_jump = estimate_jump_covariance(population, β)
        ϵ = update_epsilon(u, v)

        ## -- resample
        if n_accept >= resample

            resample_population!(population, u, δ)

            ϵ = update_epsilon(u, v)
            n_accept = 0
        end

        # update progressbar
        ProgressMeter.next!(progbar, showvalues = show_summary(state, u))
    end

    # update state
    state.ϵ .= ϵ
    state.Σ_jump = Σ_jump
    state.n_simulation += n_updates
    state.n_accept = n_accept

    @info "All particles have been $(n_simulation ÷ n_particles) times updated."
    return population_state

end


"""
```
    sabc(f_dist, prior::Distribition, args...;
                     n_particles = 100, n_simulation = 10_000,
                     eps_init::Union{Real, Vector{Real}, Tuple{Real}},
                     resample = n_particles,
                     v=1.2, β=0.8, δ=0.1,
                     kwargs...)

```
# Simulated Annealing Approximtaive Bayesian Inference Algorithm

## Arguments
- `f_dist`: Function returns distance(s) between data and a random sample from the likelihood.
            The first argument must be the parameter vector.
- `prior`: A `Distribution` defining the prior.
- `args...`: Further arguments passed to `f_dist`
- `n_particles`: Desired number of particles.
- `n_simulation`: maximal number of simulations from `f_dist`.
- `eps_init`: Initial epsilon(s)
- `v=1.2`: Tuning parameter for XXX
- `beta=0.8`: Tuning parameter for XXX
- `δ=0.1`: Tuning parameter for XXX
- `resample`: After how many accepted updates?
- `kwargs...`: Further arguments passed to `f_dist``

## Return
- An object of type `SABCresult`
"""
function sabc(f_dist::Function, prior::Distribution, args...;
              n_particles = 100, n_simulation = 10_000,
              eps_init,
              resample = n_particles,
              v=1.2, β=0.8, δ=0.1,
              kwargs...)


    ## ------------------------
    ## Initialization
    ## ------------------------

    population_state = initialization(f_dist, prior, args...;
                                      n_particles = n_particles,
                                      n_simulation = n_simulation,
                                      eps_init = eps_init,
                                      v=v, β=β,
                                      kwargs...)

    ## --------------
    ## Sampling
    ## --------------
    n_sim_remaining = n_simulation - population_state.state.n_simulation
    n_sim_remaining < n_particles && @warn "`n_simulation` to small to update all particles!"

    update_population!(population_state, f_dist, prior, args...;
                       n_simulation = n_sim_remaining,
                       v=v, β=β, δ=δ, resample=resample, kwargs...)


    return population_state
end


end
