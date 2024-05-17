module SimulatedAnnealingABC

using LinearAlgebra
using Random
import Base.show

using UnPack: @unpack
using StatsBase: mean, cov, sample, weights

using Distributions: Distribution, pdf, MvNormal, Normal
import Roots
# import ProgressMeter

using Dates
using FLoops
using FoldsThreads
using ThreadPinning
ThreadPinning.Prefs.set_os_warning(false)

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
    dist_rescale::Vector{Float64}  # vector for rescaling distances

    # Containers to store trajectories
    ϵ_history::Vector{Vector{Float64}} 
    ρ_history::Vector{Vector{Float64}}
    u_history::Vector{Vector{Float64}}

    cdfs_dist_prior              # function G in Albert et al., Statistics and Computing 25, 2015
    Σ_jump::Union{Matrix{Float64}, Float64}  # Float64 for 1d
    n_simulation::Int           # number of simulations
    n_accept::Int               # number of accepted updates
    n_resampling::Int           # number of population resamplings
end

"""
Holds results
- `population`: vector of parameter samples from the approximate posterior
- `u`: transformed distances
- `ρ`: distances
- `state`: state of algorithm
"""
struct SABCresult{T, S}
    population::Vector{T}
    u::Array{S}  # transformed distances
    ρ::Array{S}  # user-defined distances
    state::SABCstate
end

"""
Function for pretty printing

Requires 'import Base.show'
Defines a new 'show' method for object of type 'SABCresult'
Prints output when function 'sabc' returns 'SABCresult'
"""
function show(io::Base.IO, s::SABCresult)
    n_particles = length(s.population)
    mean_u = round(mean(s.u), sigdigits = 4)
    acc_rate =  round(s.state.n_accept / (s.state.n_simulation - n_particles),
                      sigdigits = 4)

    println(io, "Approximate posterior sample with $n_particles particles:")
    println(io, "  - simulations used: $(s.state.n_simulation)")
    println(io, "  - average transformed distance: $mean_u")
    println(io, "  - ϵ: $(round.(s.state.ϵ, sigdigits=4))")
    println(io, "  - population resampling: $(s.state.n_resampling)")
    println(io, "  - acceptance rate: $(acc_rate)")
    println(io, "The sample can be accessed with the field `population`.")
    println(io, "The history of ϵ can be accessed with the field `state.ϵ_history`.")
    println(io, " -------------------------------------- ")
end

# -------------------------------------------
# Algorithm functions
# -------------------------------------------
"""
Update ϵ
Single-epsilon: see eq(31) in Albert et al., Statistics and Computing 25, 2015
Multi-epsilon: new update rule
"""
function update_epsilon(u, ui, v, n)
    mean_u = mean(u, dims=1)
    mean_ui = mean_u[ui]
    if n > 1  # multiple stats
        if mean_ui <= eps()
            error("Division by zero - Mean u for statistic $ui = $mean_ui - Multi-epsilon not possible. Try single-epsilon.")
        end
        q = mean_u / mean_ui
        cn = Float64(factorial(big(2*n+2))/(factorial(big(n+1))*factorial(big(n+2))))
        num = 1 + sum(q.^(n/2))
        den = cn*(n+1)*mean_ui^(1+n/2)*prod(q.^2)
        βi = Roots.find_zero(β -> (1-exp(-β)*(1+β))/(β*(1-exp(-β))) - mean_ui, 1/mean_ui)
        ϵ_new = 1/(βi + v*num/den)      
    elseif n == 1  # one stat
        ϵ_new = mean_ui <= eps() ? zero(mean_ui) : Roots.find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - mean_ui^2, (0, mean_ui))
    else 
        error("Inconsistency - number of statistics = $n should be >= 1")
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
    idx_resampled = sample(1:n, weights(w), n, replace=true)
    
    population = population[idx_resampled]
    u = u[idx_resampled,:]
    
    # Effective sample size:
    ess = (sum(w))^2/sum(w.^2) 

    population, u, ess
end


"""
Estimate the coavariance for the jump distributions from a population
"""
function estimate_jump_covariance(population, β)
    β * cov(stack(population, dims=1)) + 1e-15*I
end


"""
Proposal for n-dimensions, n > 1
"""
proposal(θ, Σ::Matrix) = θ .+ rand(MvNormal(zeros(size(Σ,1)), Σ))

"""
Proposal for 1-dimension
"""
proposal(θ, Σ::Float64) = θ + rand(Normal(0, sqrt(Σ)))

"""
Distance
"""
dist_euclidean(d) = sqrt(sum(d.^2))


"""
# Initialisation step

## Arguments
See docs for `sabc`

type = 1 -> original single epsilon
type = 2 -> multi epsilon
type = 3 -> hybrid multiu-single-epsilon

## Value

- population
- u
- prior distances
- state
"""
function initialization(f_dist, prior::Distribution, args...;
                        n_particles, n_simulation,
                        v = 1.2, β = 0.8, δ= 0.1, type = 1, kwargs...)

    n_simulation < n_particles &&
        error("`n_simulation = $n_simulation` is too small for $n_particles particles.")

    # ------------------------------------ #
    ############ Algorithm type ############
    # 1: original single-epsilon
    # 2: multi-epsilon
    # 3: hybrid multi-u-single-epsilon 
    if type == 1
        @info "Preparing to run SABC algorithm: 'single-epsilon'"
    elseif type == 2
        @info "Preparing to run SABC algorithm: 'multi-epsilon'"
    elseif type == 3
        @info "Preparing to run SABC algorithm: 'hybrid multi-u-single-epsilon'"
    else
        error("Keyword type = $type not valid. Select type = 1, 2 or 3.")
    end

    # --------------------------------------------- #
    ############ Check available threads ############
    @info "Using threads: $(Threads.nthreads()) "; flush(stderr)
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        @info "Set BLAS threads = $(LinearAlgebra.BLAS.get_num_threads()) "; flush(stderr)
        if Sys.islinux()
            @info "Set 'pinthreads(:cores)' for optimal multi-threading performance"; flush(stderr)
            pinthreads(:cores)
        end
    end
    
    # ------------------------------------------- #
    ############ Initialize containers ############
    θ = rand(prior)  # random sample
    ρinit = f_dist(θ, args...; kwargs...)  # generate dummy distances to initialize containers 
    n_stats = length(ρinit)
    distances_prior = Array{eltype(ρinit)}(undef, n_particles, n_stats)
    distances_prior_rescaled = Array{eltype(ρinit)}(undef, n_particles, n_stats)
    if type == 1
        distances_prior_single = Array{eltype(ρinit)}(undef, n_particles, 1)
    end
    population = Vector{typeof(θ)}(undef, n_particles)

    # ---------------------------------------- #
    ############ Build prior sample ############
    @info "Initializing population..."; flush(stderr)
    for i in 1:n_particles
        ## sample
        θ = rand(prior)
        ρinit = f_dist(θ, args...; kwargs...)
        ## store parameter and distances
        population[i] = θ
        distances_prior[i,:] .= ρinit
    end
    # Keep track of the original prior distances (before rescaling)
    # We save all individual distances also for algorithm of type 1 (single-epsilon)
    ρ_history = [[mean(ic) for ic in eachcol(distances_prior)]]

    # ---------------------------------------- #
    ############ Rescale distances  ############
    dist_rescale = 1 ./ ρ_history[1]
    # Rescale all prior distances
    distances_prior_rescaled = distances_prior
    for ir in eachrow(distances_prior_rescaled)
        ir .*= dist_rescale
    end
    # and store in history
    push!(ρ_history, [mean(ic) for ic in eachcol(distances_prior_rescaled)])

    # IMPORTANT! 
    # One more thing: 
    # algorithm of type 1 (single-epsilon) needs single distance
    if type == 1
        for ix in 1:n_particles
            distances_prior_single[ix] = dist_euclidean(distances_prior_rescaled[ix,:])
        end
    end

    # --------------------------------------------- #
    ############ Learn cdf and compute ϵ ############
    # Estimate cdf of ρ under the prior
    any(distances_prior_rescaled .< 0) && error("Negative distances are not allowed!")
    if type == 1
        cdfs_dist_prior = build_cdf(distances_prior_single)
        u = similar(distances_prior_single)
        for i in 1:n_particles
            u[i,:] .= cdfs_dist_prior(distances_prior_single[i,:])
        end
    else
        cdfs_dist_prior = build_cdf(distances_prior_rescaled)
        u = similar(distances_prior_rescaled)
        # Transformed distances
        for i in 1:n_particles
            u[i,:] .= cdfs_dist_prior(distances_prior_rescaled[i,:])
        end
    end 
    
    # resampling before setting intial epsilon
    population, u, ess = resample_population(population, u, δ)
    @info "Initial resampling (δ = $δ) - ESS = $ess "  # ESS = Effective sample size
    # now, intial epsilon
    if type == 1 || type == 2
        # size(u,2) = 1 when type = 1, and size(u,2) = n_stats when type = 2
        ϵ = [update_epsilon(u, ui, v, size(u,2)) for ui in 1:size(u,2)] 
    elseif type == 3
        u_average = sum(u[:,ix] for ix in 1:n_stats)./n_stats
        ϵ = [update_epsilon(u_average, 1, v, 1)] 
    end
    
    # store
    u_history = [[mean(ic) for ic in eachcol(u)]]
    ϵ_history = [copy(ϵ)]  # needs copy() to avoid a sequence of constant values when (push!)ing 

    Σ_jump = estimate_jump_covariance(population, β)

    # -------------------------------------------------------------------- #
    ############ Collect parameters and states of the algorithm ############
    n_simulation = n_particles  # N.B.: we consider only n_particles draws from the prior 
                                # and neglect the first call to f_dist ('initialization of containers')  


    state = SABCstate(ϵ,
                      dist_rescale,
                      ϵ_history,
                      ρ_history,
                      u_history,
                      cdfs_dist_prior,
                      Σ_jump,
                      n_simulation,
                      0, 1)  # n_accept set to 0, n_resampling to 1

    @info "Population with $n_particles particles initialised."; flush(stderr)
    @info "Initial ϵ = $ϵ"; flush(stderr)
    return SABCresult(population, u, distances_prior_rescaled, state)

end


"""
Updates particles and applies importance sampling if needed. 
Modifies `population_state`.

## Arguments
See `sabc`

type = 1 -> original single epsilon
type = 2 -> multi epsilon
type = 3 -> hybrid multiu-single-epsilon
"""

function update_population!(population_state::SABCresult, f_dist, prior, args...;
                            n_simulation,
                            v=1.2, β=0.8, δ=0.1,
                            type = 1, 
                            resample=2*length(population_state.population),
                            checkpoint_history = 1,
                            checkpoint_display = 100,
                            kwargs...)
    
    state = population_state.state
    population = copy(population_state.population)
    u = copy(population_state.u)
    ρ = copy(population_state.ρ)
    n_stats = size(u,2)

    @unpack ϵ, dist_rescale, ϵ_history, ρ_history, 
            u_history, n_accept, n_resampling, Σ_jump, cdfs_dist_prior = state  
    n_particles = length(population)

    n_population_updates = n_simulation ÷ n_particles  # populations update = total simulations / particles
    n_updates = n_population_updates * n_particles     # number of calls to `f_dist`
    last_checkpoint_epsilon = 0                        # set checkpoint counter to zero

    now = Dates.now()
    inter = 0  # to estimate ETA
    if Threads.nthreads() > 1  # set basesize for parallel for loop
        # nthreads -> number of available cores
        # basesize = n_particles/nthreads -> size of the chunk assigned to each core 
        bs = ceil(Int,n_particles/Threads.nthreads())
    end
    @info "$(now) -- Starting population updates."; flush(stderr)
    

    # ---------------------------------------------------- #
    ############ Main loop (population updates) ############
    for ix in 1:n_population_updates

        if Threads.nthreads() == 1

            # ---------------------------------------------------- #
            # -- Update all particles - Single-threaded -- #
            # ---------------------------------------------------- #
            for i in eachindex(population)

                # proposal
                θproposal = proposal(population[i], Σ_jump)

                # acceptance probability
                if pdf(prior, θproposal) > 0
                    ρ_proposal = (f_dist(θproposal, args...; kwargs...)) .* dist_rescale
                    if type == 1
                        ρ_proposal_single = dist_euclidean(ρ_proposal)
                        u_proposal = cdfs_dist_prior(ρ_proposal_single)
                    else
                        u_proposal = cdfs_dist_prior(ρ_proposal)
                    end

                    accept_prob = pdf(prior, θproposal) / pdf(prior, population[i]) *
                        exp(sum((u[i,:] .- u_proposal) ./ ϵ))
                else
                    accept_prob = 0.0
                end

                if rand() < accept_prob
                    population[i] = θproposal
                    u[i,:] .= u_proposal 
                    ρ[i,:] .= ρ_proposal
                    n_accept += 1
                end

            end
            # ---------------------------------------------------- #

        elseif Threads.nthreads() > 1

            # ---------------------------------------------------- #
            # -- Update all particles - Multi-threaded -- #
            # ---------------------------------------------------- #
            # Executors: ThreadedEx() (default), TaskPoolEx(), DepthFirstEx()
            let Σ_jump = Σ_jump, ϵ = ϵ, flooccept::Int = 0
                rpopulation = Ref(population)
                ru = Ref(u)
                rρ = Ref(ρ)
                @floop ThreadedEx(basesize = bs) for i in eachindex(population)
                    # proposal
                    @inbounds θproposal = proposal(rpopulation[][i], Σ_jump)

                    # acceptance probability
                    if pdf(prior, θproposal) > 0
                        ρ_proposal = f_dist(θproposal, args...; kwargs...) .* dist_rescale
                        if type == 1
                            ρ_proposal_single = dist_euclidean(ρ_proposal)
                            u_proposal = cdfs_dist_prior(ρ_proposal_single)
                        else
                            u_proposal = cdfs_dist_prior(ρ_proposal)
                        end
                        @inbounds accept_prob = pdf(prior, θproposal) / pdf(prior, rpopulation[][i]) *
                                                exp(sum((ru[][i,:] .- u_proposal) ./ ϵ))
                    else
                        accept_prob = 0.0
                    end

                    if rand() < accept_prob
                        @inbounds rpopulation[][i] = θproposal
                        @inbounds ru[][i,:] .= u_proposal 
                        @inbounds rρ[][i,:] .= ρ_proposal
                        @reduce(flooccept += 1)
                    end
                end
                n_accept += flooccept
            end
            # ---------------------------------------------------- #
        else
            error("Unrecognized Threads.nthreads(): ", Threads.nthreads())
        end

        # ---------------------------------------------------------- #
        ############ Update epsilon and jump distribution ############
        Σ_jump = estimate_jump_covariance(population, β)

        if type == 1 || type == 2
            # size(u,2) = 1 when type = 1, and size(u,2) = n_stats when type = 2
            ϵ = [update_epsilon(u, ui, v, size(u,2)) for ui in 1:size(u,2)] 
        elseif type == 3
            u_average = sum(u[:,ix] for ix in 1:n_stats)./n_stats
            ϵ = [update_epsilon(u_average, 1, v, 1)] 
        end

        # -------------------------------- #
        ############ Resampling ############ 
        if n_accept >= (n_resampling + 1) * resample

            population, u, ess = resample_population(population, u, δ)
            Σ_jump = estimate_jump_covariance(population, β)
            if type == 1 || type == 2
                # size(u,2) = 1 when type = 1, and size(u,2) = n_stats when type = 2
                ϵ = [update_epsilon(u, ui, v, size(u,2)) for ui in 1:size(u,2)] 
            elseif type == 3
                u_average = sum(u[:,ix] for ix in 1:n_stats)./n_stats
                ϵ = [update_epsilon(u_average, 1, v, 1)] 
            end
            n_resampling += 1
            @info "Resampling $n_resampling (δ = $δ) - ESS = $ess"
        end 
       
        # ------------------------------------------------- #
        ############ Update progress and history ############
        if ix%checkpoint_display == 0
            before = now
            now = Dates.now()
            inter += (now-before).value*10^(-3)  # in seconds
            update_average_time = inter / ix 
            eta = (n_population_updates - ix) * update_average_time
            hh = lpad(floor(Int, eta/3600), 2, '0')
            mm = lpad(floor(Int, (eta % 3600)/60), 2, '0')
            ss = lpad(floor(Int, eta % 60), 2, '0')
            @info "$(now) -- Update $ix of $n_population_updates -- ETA: $(hh):$(mm):$(ss) \n ϵ: $(round.(ϵ, sigdigits=4)) \n mean transformed distance: $(round.(mean(u), sigdigits=4)) "
            flush(stderr) 
        end

        # update ϵ_history
        if ix%checkpoint_history == 0
            push!(ϵ_history, copy(ϵ))  # needs copy() to avoid a sequence of constant values
            push!(u_history, [mean(ic) for ic in eachcol(u)])
            push!(ρ_history, [mean(ic) for ic in eachcol(ρ)])
            last_checkpoint_epsilon = ix
        end
    end

    # store the last epsilon value, if not already done
    if last_checkpoint_epsilon != n_population_updates
        push!(ϵ_history, copy(ϵ))
        push!(u_history, [mean(ic) for ic in eachcol(u)])
        push!(ρ_history, [mean(ic) for ic in eachcol(ρ)])
    end

    # ---------------------------------- #
    ############ Update state ############
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

    @info "$(Dates.now())  All particles have been updated $(n_simulation ÷ n_particles) times."; flush(stderr)
    return population_state

end


"""
```
sabc(f_dist, prior::Distribition, args...;
        n_particles = 100, n_simulation = 10_000,
        type = 1,
        resample = 2*n_particles,
        v=1.2, β=0.8, δ=0.1,
        kwargs...)

```
# Simulated Annealing Approximtaive Bayesian Inference Algorithm

## Arguments
- `f_dist`: Function that distance between data and a random sample from the likelihood. The first argument must be the parameter vector.
- `prior`: A `Distribution` defining the prior.
- `args...`: Further arguments passed to `f_dist`
- `n_particles`: Desired number of particles.
- `n_simulation`: maximal number of simulations from `f_dist`.
- `v=1.2`: Tuning parameter for XXX
- `beta=0.8`: Tuning parameter for XXX
- `δ=0.1`: Tuning parameter for XXX
- type = 1 -> original single epsilon
       = 2 -> multi epsilon
       = 3 -> hybrid multiu-single-epsilon
- `resample`: After how many accepted updates?
- `kwargs...`: Further arguments passed to `f_dist``

## Return
- An object of type `SABCresult`
"""
function sabc(f_dist::Function, prior::Distribution, args...;
              n_particles = 100, n_simulation = 10_000,
              type = 1,
              resample = 2*n_particles,
              v=1.2, β=0.8, δ=0.1,
              checkpoint_history = 1,
              checkpoint_display = 100,
              kwargs...)

    # ------------------------------------ #
    ############ Initialization ############
    population_state = initialization(f_dist, prior, args...;
                                      n_particles = n_particles,
                                      n_simulation = n_simulation,
                                      v=v, β=β, δ=δ, type = type, kwargs...)

    # ------------------------------ #
    ############ Sampling ############
    n_sim_remaining = n_simulation - population_state.state.n_simulation
    n_sim_remaining < n_particles && @warn "`n_simulation` too small to update all particles!"

    update_population!(population_state, f_dist, prior, args...;
                       n_simulation = n_sim_remaining,
                       v=v, β=β, δ=δ, 
                       type = type,
                       checkpoint_history = checkpoint_history, 
                       checkpoint_display = checkpoint_display,
                       resample=resample, kwargs...)

    return population_state
end

end
