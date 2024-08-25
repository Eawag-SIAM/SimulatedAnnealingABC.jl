module SimulatedAnnealingABC

using LinearAlgebra
using Random
import Base.show

using UnPack: @unpack
using StatsBase: mean, cov, sample, weights

using Distributions: Distribution, pdf, MvNormal, Normal
import Roots

import Polyester

using Dates
using ProgressMeter

include("cdf_estimators.jl")

export sabc, update_population!

# -------------------------------------------
# Define types to hold results
# -------------------------------------------
"""
Holds state of algorithm
"""
mutable struct SABCstate
    ϵ::Vector{Float64}             # epsilon = temperature
    type::Symbol                   # the algorithm used

    # Containers to store trajectories
    ϵ_history::Vector{Vector{Float64}}
    ρ_history::Vector{Vector{Float64}}
    u_history::Vector{Vector{Float64}}

    cdfs_dist_prior             # function G in Albert et al., Statistics and Computing 25, 2015
    Σ_jump::Union{Matrix{Float64}, Float64}  # Float64 for 1d
    n_simulation::Int           # number of simulations
    n_accept::Int               # number of accepted updates
    n_resampling::Int           # number of population resamplings
end

"""
Holds results from a SABC run with fields:
- `population`: vector of parameter samples from the approximate posterior
- `u`: transformed distances
- `ρ`: distances
- `state`: state of algorithm

The history of ϵ can be accessed with the field `state.ϵ_history`.
"""
struct SABCresult{T, S}
    population::Vector{T}
    u::Array{S}  # transformed distances
    ρ::Array{S}  # user-defined distances
    state::SABCstate
end

"""
Function for pretty printing
"""
function show(io::Base.IO, s::SABCresult)
    n_particles = length(s.population)
    mean_u = round(mean(s.u), sigdigits = 4)
    acc_rate =  round(s.state.n_accept / (s.state.n_simulation - n_particles),
                      sigdigits = 4)

    println(io, "Approximate posterior sample with $n_particles particles:")
    println(io, "  - type: :$(s.state.type)")
    println(io, "  - simulations used: $(s.state.n_simulation)")
    println(io, "  - average transformed distance: $mean_u")
    println(io, "  - ϵ: $(round.(s.state.ϵ, sigdigits=4))")
    println(io, "  - population resampling: $(s.state.n_resampling)")
    println(io, "  - acceptance rate: $(acc_rate)")
    println(io, "The sample can be accessed with the field `population`.")
    println(io, "The history of ϵ can be accessed with the field `state.ϵ_history`.")
end

# -------------------------------------------
# Algorithm functions
# -------------------------------------------


"""
Update ϵ for a single statistics. See eq(31) in Albert et al., Statistics and Computing 25, 2015
"""
function update_epsilon_single_stat(ū, v)

    ϵ_new = ū <= eps() ? zero(ū) : Roots.find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - ū^2, (0, ū))
    Float64[ϵ_new]
end

"""
Update ϵ
Multi-epsilon:  new update rule
"""
function update_epsilon_multi_stats(u, v)
    n = size(u, 2)        # number of statistics
    ū = mean(u, dims=1)
    cn = Float64(factorial(big(2*n+2))/(factorial(big(n+1))*factorial(big(n+2))))
    ϵ_new = Vector{Float64}(undef, n)
    for i in 1:n
        ūi = ū[i]
        if ūi <= eps()
            error("Division by zero - Mean u for statistic $i = $ūi")
        end
        q = ū ./ ūi
        num = 1 + sum(q.^(n/2))
        den = cn*(n+1)*ūi^(1+n/2)*prod(q)
        βi = Roots.find_zero(β -> (1-exp(-β)*(1+β))/(β*(1-exp(-β))) - ūi, 1/ūi)
        ϵ_new[i] = 1/(βi + v*num/den)
    end
    ϵ_new
end



"""
Resample population
"""
function resample_population(population, u, δ)

    n = length(population)
    u_means = mean(u, dims=1)
    w = exp.(-sum(u[:,i] .* δ ./ u_means[i] for i in 1:size(u, 2)))
    # Choose indexes based on weights w
    idx_resampled = sample(1:n, weights(w), n, replace=true)
    # Apply selected indexes to population and u's
    population = population[idx_resampled]
    u = u[idx_resampled,:]
    # Effective sample size:
    ess = (sum(w))^2/sum(abs2, w)
    # Return:
    population, u, ess
end


"""
Estimate the covariance for the jump distributions from a population
"""
function estimate_jump_covariance(population, β)
    β * (cov(stack(population, dims=1)) + 1e-8*I)
end

"""
Proposal for n-dimensions, n > 1
"""
proposal(θ, Σ::AbstractArray) = θ .+ rand(MvNormal(zeros(size(Σ,1)), Σ))

"""
Proposal for 1-dimension
"""
proposal(θ, Σ::Float64) = θ + rand(Normal(0, sqrt(Σ)))


"""
# Initialisation step

## Arguments
See docs for `sabc`

## Return
- An object of type `SABCresult`
"""
function initialization(f_dist, prior::Distribution, args...;
                        n_particles, n_simulation,
                        v = 1.0, β = 0.8, δ= 0.1, type = :multi, kwargs...)

    n_simulation < n_particles &&
        error("`n_simulation = $n_simulation` is too small for $n_particles particles.")

    @info "Initialization with '$(type)-ϵ'"

    # ---------------------
    # Initialize containers

    θ = rand(prior)  # take a random sample
    ρinit = f_dist(θ, args...; kwargs...)  # generate dummy distances to initialize containers
    n_stats = length(ρinit)
    distances_prior = Array{eltype(ρinit)}(undef, n_particles, n_stats)
    population = Vector{typeof(θ)}(undef, n_particles)

    # ------------------
    # Build prior sample

    Polyester.@batch for i in 1:n_particles
        ## sample
        θ = rand(prior)
        ρinit = f_dist(θ, args...; kwargs...)
        ## store parameter and distances
        population[i] = θ
        distances_prior[i,:] .= ρinit
    end
    ρ_history = [[mean(ic) for ic in eachcol(distances_prior)]]


    # ------------------
    # Estimate the cdf of ρ under the prior

    any(distances_prior .< 0) && error("Negative distances are not allowed!")

    cdfs_dist_prior = build_cdf(distances_prior)
    u = similar(distances_prior)
    # Transformed distances
    for i in 1:n_particles
        u[i,:] .= cdfs_dist_prior(distances_prior[i,:])
    end


    # ------------------
    # resampling before setting intial epsilon
    population, u, ess = resample_population(population, u, δ)

    # initialize epsilon
    if type == :multi
        ϵ = update_epsilon_multi_stats(u, v)
    elseif  type == :hybrid
        ϵ = update_epsilon_single_stat(mean(u), v)
    end

    # store
    u_history = [[mean(ic) for ic in eachcol(u)]]
    ϵ_history = [copy(ϵ)]

    Σ_jump = estimate_jump_covariance(population, β)

    # ---------------------
    # Collect parameters and states of the algorithm

    n_simulation = n_particles  # N.B.: we consider only n_particles draws from the prior
                                # and neglect the first call to f_dist ('initialization of containers')

    state = SABCstate(ϵ,
                      type,
                      ϵ_history,
                      ρ_history,
                      u_history,
                      cdfs_dist_prior,
                      Σ_jump,
                      n_simulation,
                      0, 1)  # n_accept set to 0, n_resampling to 1

    return SABCresult(population, u, distances_prior, state)

end


"""
```
update_population!(population_state::SABCresult,
                   f_dist, prior, args...;
                   n_simulation,
                   v=1.0, β=0.8, δ=0.1,
                   resample = 2*length(population_state.population),
                   checkpoint_history = 1,
                   show_progressbar::Bool = !is_logging(stderr),
                   show_checkpoint = is_logging(stderr) ? 100 : Inf,
                   kwargs...)
```

Updates particles with `n_simulation` and applies importance sampling if needed.
Modifies `population_state`.

## Arguments
See docstring for `sabc`.

"""
function update_population!(population_state::SABCresult, f_dist, prior, args...;
                            n_simulation,
                            v=1.0, β=0.8, δ=0.1,
                            resample = 2*length(population_state.population),
                            checkpoint_history = 1,
                            show_progressbar::Bool = !is_logging(stderr),
                            show_checkpoint = is_logging(stderr) ? 100 : Inf,
                            kwargs...)

    state = population_state.state
    population = copy(population_state.population)
    u = copy(population_state.u)
    ρ = copy(population_state.ρ)
    n_stats = size(u,2)

    @unpack ϵ, type, ϵ_history, ρ_history,
    u_history, n_accept, n_resampling, Σ_jump, cdfs_dist_prior = state

    n_particles = length(population)

    n_population_updates = n_simulation ÷ n_particles  # populations update = total simulations / particles
    n_updates = n_population_updates * n_particles     # number of calls to `f_dist`
    last_checkpoint_epsilon = 0                        # set checkpoint counter to zero

    # to estimate ETA
    t_start = Dates.now()

    # ----------------------------------------------------
    #  Update all particles

    pmeter = Progress(n_population_updates; desc = "$n_population_updates population updates:",
                      output = stderr, enabled = show_progressbar)
    generate_showvalues(ϵ, u) = () -> [("ϵ", round.(ϵ, sigdigits=4)), ("average transformed distance", round.(mean(u), sigdigits=4))]
    for ix in 1:n_population_updates

        # ----------------------------------------------------------
        # update particles

        Polyester.@batch reduction = ((+, n_accept), ) for i in eachindex(population)

            # proposal
            θproposal = proposal(population[i], Σ_jump)

            # acceptance probability
            if pdf(prior, θproposal) > 0
                ρ_proposal = f_dist(θproposal, args...; kwargs...)
                u_proposal = cdfs_dist_prior(ρ_proposal)

                accept_prob = pdf(prior, θproposal) / pdf(prior, population[i]) *
                    exp(sum((u[i,:] .- u_proposal) ./ ϵ))
            else
                accept_prob = 0.0
            end

            if rand() < accept_prob
                population[i] = θproposal
                u[i,:] .= u_proposal
                ρ[i,:] .= ρ_proposal
                n_accept +=1
            end

        end


        # ----------------------------------------------------------
        # Update epsilon and jump distribution

        Σ_jump = estimate_jump_covariance(population, β)

        if type == :multi
            ϵ = update_epsilon_multi_stats(u, v)
        elseif type == :hybrid
            ϵ = update_epsilon_single_stat(mean(u), v)
        end

        # --------------------------------
        # Resampling

        if n_accept >= (n_resampling + 1) * resample

            population, u, ess = resample_population(population, u, δ)
            Σ_jump = estimate_jump_covariance(population, β)

            # update epsilon
            if type == :multi
                ϵ = update_epsilon_multi_stats(u, v)
            elseif type == :hybrid
                ϵ = update_epsilon_single_stat(mean(u), v)
            end

            n_resampling += 1
        end

        # -------------------------------------------------
        # Update progress and history

        if ix % show_checkpoint == 0
            eta = ((Dates.now() - t_start) ÷ ix) * (n_population_updates - ix)
            etastr = eta > Dates.Second(1) ? Dates.canonicalize(round(eta, Dates.Second)) : "< 1 Second"
            @info "Update $ix of $n_population_updates. average transformed distance: $(round.(mean(u), sigdigits=4)), " *
                "ϵ: $(round.(ϵ, sigdigits=4)), ETA: $(etastr)"
        end

        # update ϵ_history
        if ix % checkpoint_history == 0
            push!(ϵ_history, copy(ϵ))  # needs copy() to avoid a sequence of constant values
            push!(u_history, [mean(ic) for ic in eachcol(u)])
            push!(ρ_history, [mean(ic) for ic in eachcol(ρ)])
            last_checkpoint_epsilon = ix
        end

        next!(pmeter, showvalues = generate_showvalues(ϵ, u))
    end

    # store the last epsilon value, if not already done
    if last_checkpoint_epsilon != n_population_updates
        push!(ϵ_history, copy(ϵ))
        push!(u_history, [mean(ic) for ic in eachcol(u)])
        push!(ρ_history, [mean(ic) for ic in eachcol(ρ)])
    end

    # ----------------------------------
    # Update state

    state.ϵ = ϵ
    state.ϵ_history = ϵ_history
    state.u_history = u_history
    state.ρ_history = ρ_history
    state.Σ_jump = Σ_jump
    state.n_simulation += n_updates
    state.n_accept = n_accept
    state.n_resampling = n_resampling
    population_state.population .= population
    population_state.u .= u
    population_state.ρ .= ρ

    @info "All particles have been updated $(n_simulation ÷ n_particles) times."
    return population_state

end


"""
```
sabc(f_dist::Function, prior::Distribution, args...;
      n_particles = 100, n_simulation = 10_000,
      type = :multi,
      resample = 2*n_particles,
      v=1.0, β=0.8, δ=0.1,
      checkpoint_history = 1,
      show_progressbar::Bool = !is_logging(stderr),
      show_checkpoint = is_logging(stderr) ? 100 : Inf,
      kwargs...)
```
# Simulated Annealing Approximate Bayesian Inference Algorithm

## Arguments
- `f_dist`: Function that one or more distances between data and a random sample from the likelihood. The first argument must be the parameter vector.
- `prior`: A `Distribution` defining the prior.
- `args...`: Further arguments passed to `f_dist`
- `n_particles`: Desired number of particles.
- `n_simulation`: maximal number of simulations from `f_dist`.
- `v = 1.0`: Tuning parameter for XXX
- `β = 0.8`: Tuning parameter for XXX
- `δ = 0.1`: Tuning parameter for XXX
- `type = :multi`: Choose algorithm, either `:multi`, or `:hybrid`
- `resample`: After how many accepted population updates?
- `checkpoint_history = 1`: every how many population updates distances and epsilons are stored
- `show_progressbar::Bool = !is_logging(stderr)`: defaults to `true` for interactive use.
- `show_checkpoint::Int = 100`: every how many population updates algorithm state is displayed.
                                By default disabled for for interactive use.
- `kwargs...`: Further arguments passed to `f_dist``

## Return
- An object of type `SABCresult`
"""
function sabc(f_dist::Function, prior::Distribution, args...;
              n_particles = 100, n_simulation = 10_000,
              type = :multi,
              resample = 2*n_particles,
              v=1.0, β=0.8, δ=0.1,
              checkpoint_history = 1,
              show_progressbar::Bool = !is_logging(stderr),
              show_checkpoint = is_logging(stderr) ? 100 : Inf,
              kwargs...)

    if !(type == :multi || type == :hybrid)
        error("""Argument `type` must be :multi or :hybrid, not `$type`!""")
    end


    # ------------------------------------
    # Initialization

    population_state = initialization(f_dist, prior, args...;
                                      n_particles = n_particles,
                                      n_simulation = n_simulation,
                                      v=v, β=β, δ=δ, type = type, kwargs...)

    # ------------------------------
    # Sampling

    n_sim_remaining = n_simulation - population_state.state.n_simulation
    n_sim_remaining < n_particles && @warn "`n_simulation` too small to update all particles!"

    update_population!(population_state, f_dist, prior, args...;
                       n_simulation = n_sim_remaining,
                       resample = resample,
                       v=v, β=β, δ=δ,
                       checkpoint_history = checkpoint_history,
                       show_progressbar = show_progressbar,
                       show_checkpoint = show_checkpoint,
                       kwargs...)

    return population_state
end


# -------------------------------------------
# Helpers
# -------------------------------------------

# Check if a stream is logged. From: https://github.com/timholy/ProgressMeter.jl
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

end
