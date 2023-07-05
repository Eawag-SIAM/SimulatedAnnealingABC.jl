module SimulatedAnnealingABC


using LinearAlgebra
using Random

using UnPack: @unpack
using StatsBase: mean, cov, ecdf, sample, weights
using Distributions: Distribution, pdf, MvNormal
import Roots
import ProgressMeter

export sabc

"""
Solve for ϵ

See eq(31)
"""
function update_epsilon(u, v)
    mean_u = mean(u)
    ϵ_new = mean_u <= eps() ? zero(mean_u) : Roots.find_zero(ϵ -> ϵ^2 + v * ϵ^(3/2) - mean_u^2, (0, mean_u))
    ϵ_new, mean_u
end


"""
Resample population

"""
function resample_population!(samples_posterior, u_posterior, mean_u, δ)
    n = length(samples_posterior)
    w = exp.(-u_posterior .* δ ./ mean_u)
    idx_resampled = sample(1:n, weights(w), n, replace=true)

    permute!(samples_posterior, idx_resampled)
    permute!(u_posterior, idx_resampled)

    @info "Resampling. Effective sample size: $(round(1/sum(abs2, w ./ sum(w)), digits=2))"
end


"""
Estimate the coavariance for the jump distributions from an population
"""
function estimate_jump_covariance(samples_posterior, β)
    β * cov(stack(samples_posterior, dims=1)) + 1e-6*I
end

"""
# Initialisation step

## # Arguments
See docs for `sabc`.
"""
function initialization_noninf(f_dist, prior::Distribution, args...;
                               n_particles, n_simulation, eps_init,
                               v=0.3, β=1.0, kwargs...)

    ## ------------------------
    ## Initialize containers

    θ = rand(prior)
    samples_posterior = Vector{typeof(θ)}(undef, n_particles)
    dist_posterior = Vector{Float64}(undef, n_particles)
    dist_prior = Float64[]


    ## ------------------
    ## Build prior sample

    iter = 0
    counter = 0 # Number of accepted particles in samples_posterior
    progbar = ProgressMeter.Progress(n_particles, desc="Generating initial population...", dt=0.5)
    while counter < n_particles

        iter += 1
        if iter > n_simulation
            error("'n_simulation' reached! The initial sample could not be generated.")
        end

        ## Generate new particle
        θ = rand(prior)
        ρ = f_dist(θ, args...; kwargs...)

        ## store distance
        push!(dist_prior, ρ)

        ## Accept with Prob = exp(-rho.p/eps.init) and store in samples_posterior
        if rand() < exp(-ρ/eps_init)
            counter += 1
            samples_posterior[counter] = θ
            dist_posterior[counter] = ρ
            ProgressMeter.next!(progbar)
        end
    end

    ## ----------------------------------
    ## Compute ϵ

    ## empirical cdf of ρ under the prior
    cdf_G = ecdf(dist_prior)

    u_posterior = cdf_G(dist_posterior)

    ϵ, mean_u = update_epsilon(u_posterior, v)
    Σ_jump = estimate_jump_covariance(samples_posterior, β)

    n_simulation = length(dist_prior)
    @info "Ensample with $n_particles particles initialised with $n_simulation simulation."

    return (samples_posterior=samples_posterior, u_posterior=u_posterior,
            mean_u=mean_u, ϵ=ϵ, cdf_G=cdf_G, Σ_jump=Σ_jump,
            n_simulation_init = n_simulation)

end



"""
```
    sabc(f_dist, prior::Distribition, args...;
                     n_particles = 100, n_simulation = 10_000,
                     eps_init = 1.0,
                     resample = n_particles,
                     v=0.3, β=1.0, δ=0.9,
                     adaptjump = true,
                     kwargs...)

```
# Simulated Annealing Approximtaive Bayesian Inference Algorithm

## Arguments
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

## Returns
- A named tuple with the following keys:
    - `posterior_samples`
    - `eps`: value of epsilon at the last iteration.
"""
function sabc(f_dist, prior::Distribution, args...;
              n_particles = 100, n_simulation = 10_000,
              eps_init,
              resample = n_particles,
              v=0.3, β=1.5, δ=0.9,
              adaptjump = true,
              kwargs...)


    ## ------------------------
    ## Initialize
    ## ------------------------

    @unpack (samples_posterior, u_posterior, mean_u, ϵ, Σ_jump, cdf_G, n_simulation_init) =
        initialization_noninf(f_dist, prior, args...;
                              n_particles = n_particles,
                              n_simulation = n_simulation,
                              eps_init = eps_init,
                              v=v, β=β,
                              kwargs...)

    dim_par = size(prior)


    ## --------------
    ## Sampling
    ## --------------

    progbar = ProgressMeter.Progress(n_simulation, desc="Sampling...", dt=0.5)
    show_summary(ϵ, mean_u) = () -> [(:eps, ϵ), (:mean_transformed_distance, mean_u)]

    n_accept = 0
    for _ in 1:(n_simulation - n_simulation_init)

        ## -- update all particles (this can be multithreaded)
        for i in 1:n_particles

            # proposal
            θproposal = samples_posterior[i] .+ rand(MvNormal(zeros(dim_par), Σ_jump))

            # acceptance probability
            if pdf(prior, θproposal) > 0
                u_proposal = cdf_G(f_dist(θproposal, args...; kwargs...))
                accept_prob = pdf(prior, θproposal) / pdf(prior, samples_posterior[i]) * exp((u_posterior[i] - u_proposal) / ϵ)
            else
                accept_prob = 0.0
            end

            # if isnan(accept_prob)
            #     accept_prob = u_posterior[i] < u_proposal ? 0.0 : 1.0
            # end

            if rand() < accept_prob
                samples_posterior[i] = θproposal
                u_posterior[i] = u_proposal
                n_accept += 1
            end

        end

        ## -- update epsilon and jump distribution
        if adaptjump
            Σ_jump = estimate_jump_covariance(samples_posterior, β)
        end

        ϵ, mean_u = update_epsilon(u_posterior, v)

        ## -- resample
        if (n_accept >= resample) && (mean_u > eps())

            resample_population!(samples_posterior, u_posterior, mean_u, δ)

            ϵ, mean_u = update_epsilon(u_posterior, v)

            n_accept = 0
        end

        # update progressbar
        ProgressMeter.next!(progbar, showvalues = show_summary(ϵ, mean_u))
    end

    return (posterior_samples=samples_posterior, eps=ϵ)
end


end
