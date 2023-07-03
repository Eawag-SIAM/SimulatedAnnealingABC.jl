module SimulatedAnnealingABC


using LinearAlgebra
using Random

using UnPack: @unpack
using Roots: find_zero
using StatsBase: mean, cov, ecdf, sample, weights
using Distributions: Distribution, pdf
import ProgressMeter

export sabc


"""
Solve for ϵ

See eq(31)
"""
function schedule(ρ, v)
    if ρ <= eps(ρ)
        zero(ρ)
    else
        find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - ρ^2, (0, ρ))
    end
end

"""
Compute mean of distances WHY???
"""
function rho_mean(ϵ, dists)
    if ϵ <= eps()
        zero(eps)
    else
        (sum(exp.(-dists ./ ϵ) .* dists)) ./  sum(exp.(-dists ./ ϵ))
    end
end

"""
Resample ensample

"""
function resample_ensample!(E, dist_E, U, δ)
    n = length(E)
    w = exp.(-dist_E .* δ ./ U)
    idx_resampled = sample(1:n, weights(w), n, replace=true)

    permute!(E, idx_resampled)
    permute!(dist_E, idx_resampled)

    @info "Resampling. Effective sample size: $(round(1/sum(abs2, w ./ sum(w)), digits=2))"
end


"""
Estimate the coavariance for the jump distributions from an ensemble
"""
function estimate_jump_covariance(E, β)
    β * cov(stack(E, dims=1)) + 1e-6*I
end

"""
    Initialisation step

# Arguments
See docs for `sabc`.
"""
function initialization_noninf(f_dist, prior::Distribution, args...;
                               n_particles, n_simulation, eps_init,
                               v=0.3, β=1.0, kwargs...)

    ## ------------------------
    ## Initialize containers
    ## ------------------------

    θ = rand(prior)
    E = Vector{typeof(θ)}(undef, n_particles)
    P = typeof(θ)[]

    dist_E = Vector{Float64}(undef, n_particles)
    dist_P = Float64[]


    ## ------------------
    ## Initialization
    ## ------------------

    iter = 0
    counter = 0 # Number of accepted particles in E
    progbar = ProgressMeter.Progress(n_particles, desc="Generating initial ensemble...", dt=0.5)
    while counter < n_particles

        iter += 1
        if iter > n_simulation
            error("'n_simulation' reached! The initial sample could not be generated.")
        end

        ## Generate new particle
        θ = rand(prior)
        ρ = f_dist(θ, args...; kwargs...)

        ## store parameter in P
        push!(P, θ)
        push!(dist_P, ρ)

        ## Accept with Prob = exp(-rho.p/eps.init) and store in E
        if rand() < exp(-ρ/eps_init)
            counter += 1
            E[counter] = θ
            dist_E[counter] = ρ
            ProgressMeter.next!(progbar)
        end
    end


    ## empirical cdf of ρ under the prior
    cdf_G = ecdf(dist_P)

    dist_E = cdf_G(dist_E)
    dist_P = cdf_G(dist_P)

    ##
    U = mean(dist_E)            # CHECK! This is different in R code!
    ## or
    U = rho_mean(eps_init, dist_E)

    # γ = ... ????              # Missing in R code? Compare to paper!

    ϵ = schedule(U, v)

    Σ_jump = estimate_jump_covariance(E, β)

    return (E=E, dist_E=dist_E, P=P, U=U, ϵ=ϵ, cdf_G=cdf_G, Σ_jump=Σ_jump)

end



"""
```
    sabc(f_dist,prior::Distribition, args...;
                     n_particles = 100, n_simulation=10_000,
                     eps_init = 1.0,
                     resample = n_particles,
                     v=0.3, β=1.0, δ=0.9,
                     adaptjump = true,
                     kwargs...)

```
# Simulated Annealing Approximtaive Bayeisain Inference Algorithm

# Arguments
- `f_dist`: Function that distance between data and a random sample from the likelihood. The first argument must be the parameter vector.
- `prior`: A `Distribution` defining the prior.
- `args...`: Further arguments passed to `f_dist`
- `n_particles`: Desired number of particles.
- `n_simulation`: number of simulalations from `f_dist`.
- `eps_init`: Initial epsilon.
- `v=0.3`: Tuning parameter.
- `beta=1`: Tuning parameter.
- `δ=0.9`: Tuning parameter.
- `resample`: After how many accepted updates?
- `kwargs...`: Further arguments passed to `f_dist``

# Returns
- A named tuple with the following keys:
    - `posterior_samples`
    - `prior_samples`
    - `eps`: value of epsilon at the last iteration.
"""
function sabc(f_dist, prior::Distribution, args...;
              n_particles = 100, n_simulation = 10_000,
              eps_init,
              resample = n_particles, # CHECK! meaningful default?
              v=0.3, β=1.5, δ=0.9,
              adaptjump = true,
              kwargs...)


    ## ------------------------
    ## Initialize
    ## ------------------------

    @unpack E, dist_E, U, ϵ, Σ_jump, cdf_G, P = initialization_noninf(f_dist, prior, args...;
                                                                      n_particles = n_particles,
                                                                      n_simulation = n_simulation,
                                                                      eps_init = eps_init,
                                                                      v=v, β=β,
                                                                      kwargs...)

    dim_par = size(prior)
    n_simulation_init = length(P)
    @info "Ensample with $n_particles particles initialised with $n_simulation_init simulation."

    ## --------------
    ## Sampling
    ## --------------

    progbar = ProgressMeter.Progress(n_simulation, desc="Sampling...", dt=0.5)
    show_summary(ϵ, U) = () -> [(:eps, ϵ), (:mean_distance, U)]

    accept = 0
    for _ in 1:(n_simulation - n_simulation_init)

        idx = rand(1:n_particles)  # pick a particle

        # proposal
        θproposal = E[idx] .+ rand(Distributions.MvNormal(zeros(dim_par), Σ_jump))
        if pdf(prior, θproposal) > 0
            dist_proposal = cdf_G(f_dist(θproposal, args...; kwargs...))
            accept_prob = pdf(prior, θproposal) / pdf(prior, E[idx]) * exp((dist_E[idx] - dist_proposal) / ϵ)
        else
            accept_prob = 0.0
        end

        # if isnan(accept_prob)
        #     accept_prob = ifelse(dist_E[idx] < dist_proposal, 0.0, 1.0)
        # end

        if rand() < accept_prob
            E[idx] = θproposal
            dist_E[idx] = dist_proposal

            if adaptjump
                Σ_jump = estimate_jump_covariance(E, β) # Do we want to do this every time?
             end

             U = mean(dist_E)
             ϵ = schedule(U, v)
             accept += 1
        end


        # resample
        if (accept >= resample) && (U > eps())

            resample_ensample!(E, dist_E, U, δ)


            ϵ = ϵ * (1 - δ)
            U = mean(dist_E)            # WHY?
            ϵ = schedule(U, v)          # WHY? should be no change!

            accept = 0
        end

        # update progressbar
        ProgressMeter.next!(progbar, showvalues = show_summary(ϵ, U))
    end

    return (posterior_samples=E, prior_samples=P, eps=ϵ)
end


end
