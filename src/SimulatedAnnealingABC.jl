module SimulatedAnnealingABC

using LinearAlgebra
using Random
import Base.show

using UnPack: @unpack
using StatsBase: mean, cov, sample, weights

using Distributions: Distribution, pdf, MvNormal, Normal
import Roots

import Dates
using ProgressMeter

include("cdf_estimators.jl")
include("proposals.jl")

export sabc, update_population!


# -------------------------------------------
# Define types to hold results
# -------------------------------------------
"""
Holds state of algorithm
"""
mutable struct SABCstate
    ϵ::Vector{Float64}             # epsilon = temperature
    algorithm::Symbol              # the algorithm used

    # Containers to store trajectories
    ϵ_history::Vector{Vector{Float64}}
    ρ_history::Vector{Vector{Float64}}
    u_history::Vector{Vector{Float64}}

    cdfs_dist_prior             # function G in Albert et al., Statistics and Computing 25, 2015
    n_simulation::Int           # number of simulations
    n_accept::Int               # number of accepted updates
    n_resampling::Int           # number of population resamplings
    n_population_updates::Int   # number of population updates
end

"""
Holds results from a SABC run with fields:
- `population`: vector of parameter samples from the approximate posterior
- `u`: transformed distances
- `ρ`: distances
- `state`: state of algorithm

The history of ϵ can be accessed with the field `state.ϵ_history`.
The history of ρ can be accessed with the field `state.ρ_history`.
The history of u can be accessed with the field `state.u_history`.
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
    println(io, "  - algorithm: :$(s.state.algorithm)")
    println(io, "  - simulations used: $(s.state.n_simulation)")
    println(io, "  - number of population updates: $(s.state.n_population_updates)")
    println(io, "  - average transformed distance: $mean_u")
    println(io, "  - ϵ: $(round.(s.state.ϵ, sigdigits=4))")
    println(io, "  - number of population resamplings: $(s.state.n_resampling)")
    println(io, "  - acceptance rate: $(acc_rate)")
    println(io, "The sample can be accessed with the field `population`.")
    println(io, "The history of ϵ can be accessed with the field `state.ϵ_history`.")
    println(io, "The history of ρ can be accessed with the field `state.ρ_history`.")
    println(io, "The history of u can be accessed with the field `state.u_history`.")
end

# -------------------------------------------
# Algorithm functions
# -------------------------------------------


"""
Update a single ϵ. See eq(31) in Albert et al., Statistics and Computing 25, 2015
"""
function update_epsilon_single_eps(ū, v)
    ϵ_new = ū <= eps() ? zero(ū) : Roots.find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - ū^2, (0, ū))
    Float64[ϵ_new]
end

"""
Update multible ϵ. See eq(19-20) in Albert et al. (in preparation)
"""
function update_epsilon_multi_eps(u, v)
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
    ū = mean(u, dims=1)
    w = exp.(-sum(u[:,i] .* δ ./ ū[i] for i in 1:size(u, 2)))
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
# Initialization step

## Arguments
See docs for `sabc`

## Returns
- An object of type `SABCresult`
"""
function initialization(f_dist, prior::Distribution, args...;
                        n_particles, n_simulation,
                        v = 1.0, δ= 0.1, algorithm = :single_eps, kwargs...)

    n_simulation < n_particles &&
        error("`n_simulation = $n_simulation` is too small for $n_particles particles.")

    @info "Initialization for '$(algorithm)'"; flush(stderr)

    # ---------------------
    # Initialize containers

    θ = rand(prior)  # take a random sample
    ρinit = f_dist(θ, args...; kwargs...)  # generate dummy distances to initialize containers
    n_stats = length(ρinit)
    distances_prior = Array{eltype(ρinit)}(undef, n_particles, n_stats)
    population = Vector{typeof(θ)}(undef, n_particles)

    # ------------------
    # Build prior sample

    Threads.@threads for i in 1:n_particles
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
    if algorithm == :multi_eps
        ϵ = update_epsilon_multi_eps(u, v)
    elseif  algorithm == :single_eps
        ϵ = update_epsilon_single_eps(mean(u), v)
    end

    # store
    u_history = [[mean(ic) for ic in eachcol(u)]]
    ϵ_history = [copy(ϵ)]

    # ---------------------
    # Collect parameters and state of the algorithm

    n_simulation = n_particles  # N.B.: we consider only n_particles draws from the prior
                                # and neglect the first call to f_dist ('initialization of containers')

    state = SABCstate(ϵ,
                      algorithm,
                      ϵ_history,
                      ρ_history,
                      u_history,
                      cdfs_dist_prior,
                      n_simulation,
                      0, 1, 0)  # n_accept = 0, n_resampling = 1, n_population_updates = 0

    return SABCresult(population, u, distances_prior, state)

end


"""
```
update_population!(population_state::SABCresult,
                   f_dist, prior, args...;
                   n_simulation,
                   v=1.0, δ=0.1,
                   proposal::Proposal = DifferentialEvolution(n_para = length(prior)),
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
                            v=1.0, δ=0.1,
                            proposal::Proposal = DifferentialEvolution(n_para = length(prior)),
                            resample = 2*length(population_state.population),
                            checkpoint_history = 1,
                            show_progressbar::Bool = !is_logging(stderr),
                            show_checkpoint = is_logging(stderr) ? 100 : Inf,
                            kwargs...)

    v <= 0 && error("Annealing speed `v` must be positive.")
    δ <= 0 && error("Resamping intensity `δ` must be positive.")

    state = population_state.state
    population = copy(population_state.population)
    u = copy(population_state.u)
    ρ = copy(population_state.ρ)
    n_stats = size(u,2)

    @unpack ϵ, algorithm, ϵ_history, ρ_history,
    u_history, n_accept, n_resampling, cdfs_dist_prior = state

    n_particles = length(population)

    n_population_updates = n_simulation ÷ n_particles  # population updates = total simulations / particles
    n_updates = n_population_updates * n_particles     # number of calls to `f_dist` (individual particle updates)
    last_checkpoint_epsilon = 0                        # set checkpoint counter to zero

    # to estimate ETA
    t_start = Dates.now()

    # ------------------
    # estimate jump covariance
    update_proposal!(proposal, population)


    # ----------------------------------------------------
    #  Update all particles

    pmeter = Progress(n_population_updates; desc = "$n_population_updates population updates:",
                      output = stderr, enabled = show_progressbar)
    generate_showvalues(ϵ, u) = () -> [("ϵ", round.(ϵ, sigdigits=4)), ("average transformed distance", round.(mean(u), sigdigits=4))]

    for ix in 1:n_population_updates

        # ----------------------------------------------------------
        # update particles

        # split population indices in two halves
        batch_1 = 1:(n_particles÷2)
        batch_2 = (n_particles÷2 + 1):n_particles

        n_accept_tmp = Threads.Atomic{Int}(0)
        for (active, inactive) in [(batch_1, batch_2), (batch_2, batch_1)]

            population_inactive = @view population[inactive]

            Threads.@threads for i in active

                # generate proposal
                θproposal = proposal(population[i],  population_inactive)

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
                    Threads.atomic_add!(n_accept_tmp, 1)
                end

            end
        end

        n_accept += n_accept_tmp[]


        # --------------------------------
        # Resampling

        if n_accept >= (n_resampling + 1) * resample
            population, u, ess = resample_population(population, u, δ)
            n_resampling += 1
        end

        # ----------------------------------------------------------
        # Update epsilon and proposal distribution

        update_proposal!(proposal, population)

        if algorithm == :multi_eps
            ϵ = update_epsilon_multi_eps(u, v)
        elseif algorithm == :single_eps
            ϵ = update_epsilon_single_eps(mean(u), v)
        end

        # -------------------------------------------------
        # Update progress and history

        if ix % show_checkpoint == 0
            eta = ((Dates.now() - t_start) ÷ ix) * (n_population_updates - ix)
            etastr = eta > Dates.Second(1) ? Dates.canonicalize(round(eta, Dates.Second)) : "< 1 Second"
            @info "Update $ix of $n_population_updates. Average transformed distance: $(round.(mean(u), sigdigits=4)), " *
                "ϵ: $(round.(ϵ, sigdigits=4)), ETA: $(etastr)"; flush(stderr)
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
    state.n_simulation += n_updates
    state.n_accept = n_accept
    state.n_resampling = n_resampling
    state.n_population_updates += n_population_updates
    population_state.population .= population
    population_state.u .= u
    population_state.ρ .= ρ

    @info "All particles have been updated $(n_population_updates) times."; flush(stderr)
    return population_state

end


"""
```
sabc(f_dist::Function, prior::Distribution, args...;
      n_particles = 100, n_simulation = 10_000,
      algorithm = :single_eps,
      propsal =  DifferentialEvolution(γ0=2.38/sqrt(2*n_para)).
      resample = 2*n_particles,
      v=1.0, δ=0.1,
      checkpoint_history = 1,
      show_progressbar::Bool = !is_logging(stderr),
      show_checkpoint = is_logging(stderr) ? 100 : Inf,
      kwargs...)
```
# Simulated Annealing Approximate Bayesian Inference Algorithm

## Arguments
- `f_dist`: Function that returns one or more distances between data and a random sample from the likelihood. The first argument must be the parameter vector.
- `prior`: A `Distribution` defining the prior.
- `args...`: Further arguments passed to `f_dist`
- `n_particles`: Desired number of particles.
- `n_simulation`: maximal number of simulations from `f_dist`.
- `algorithm = :single_eps`: Choose algorithm, either `:multi_eps`, or `:single_eps`. With `:single_eps` a global tolerance is used for all distances. Wit `:multi_eps` every distnace has it's own tolerance.
- `propsal =  DifferentialEvolution(n_para = length(prior))`: Method to generate propsals. Currently `RandomWalk`, `DifferentialEvolution`, and `StretchMove` are implemented.
- `v = 1.0`: Tuning parameter for annealing speed. Must be positive.
- `δ = 0.1`: Tuning parameter for resampling intensity. Must be positive and should be small.
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
              algorithm = :single_eps,
              proposal::Proposal = DifferentialEvolution(n_para = length(prior)),
              resample = 2*n_particles,
              v=1.0, δ=0.1,
              checkpoint_history = 1,
              show_progressbar::Bool = !is_logging(stderr),
              show_checkpoint = is_logging(stderr) ? 100 : Inf,
              kwargs...)

    if !(algorithm == :multi_eps || algorithm == :single_eps)
        error("""Argument `algorithm` must be :multi_eps or :single_eps, not `$algorithm`!""")
    end


    # ------------------------------------
    # Initialization

    population_state = initialization(f_dist, prior, args...;
                                      n_particles = n_particles,
                                      n_simulation = n_simulation,
                                      v=v,  δ=δ, algorithm = algorithm, kwargs...)

    # ------------------------------
    # Sampling

    n_sim_remaining = n_simulation - population_state.state.n_simulation
    n_sim_remaining < n_particles && @warn "`n_simulation` too small to update all particles!"

    update_population!(population_state, f_dist, prior, args...;
                       n_simulation = n_sim_remaining,
                       resample = resample,
                       proposal = proposal,
                       v=v, δ=δ,
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
