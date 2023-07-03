module SimulatedAnnealingABC


using LinearAlgebra
using Random

using UnPack: @unpack
using Roots: find_zero
using StatsBase: mean, cov, ecdf, sample, weights
import Distributions
import ProgressMeter

export sabc_noninf


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
    if ϵ <= eps(ϵ)
        zero(eps)
    else
        (sum(exp.(-dists ./ ϵ) .* dists)) ./  sum(exp.(-dists ./ ϵ))
    end
end


"""
    Initialisation step

# Arguments
- `f_dist`: Function that  returns a the distance between data and a random sample from the likelhood. The first argument must be the parameter vector.
- `d_prior`: Function that returns the density of the prior distribution.
- `r_prior`: Function which returns one vector of a random realization of the prior distribution.
- `n_sample`: Desired number of samples.
- `eps_init`: Initial epsilon.
- `iter_max`: Maximal number of iterations.
- `v`: Tuning parameter.
- `beta`: Tuning parameter.
- `args...`: Further arguments passed to f_dist, aside of the parameter.

"""
function initialization_noninf(f_dist, r_prior, args...; n_sample,
                               eps_init=1.0, iter_max=10_000,
                               v=0.3, β=1.0, kwargs...)

    ## ------------------------
    ## Initialize containers
    ## ------------------------

    θ = r_prior()
    dim_par = length(θ)

    E = Vector{typeof(θ)}(undef, n_sample)
    P = typeof(θ)[]

    dist_E = Vector{Float64}(undef, n_sample)
    dist_P = Float64[]


    ## ------------------
    ## Initialization
    ## ------------------

    iter = 0
    counter = 0 # Number of accepted particles in E
    progbar = ProgressMeter.Progress(n_sample, desc="Generating initial ensemble...", dt=0.5)
    while counter < n_sample

        iter += 1
        if iter > iter_max
            error("'iter_max' reached! No initial sample could be generated.")
        end

        ## Generate new particle
        θ = r_prior()
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
    U = mean(dist_E)            # CHECK!
    ## or
    U = rho_mean(eps_init, dist_E)

    # γ = ... ????              # Missing in R code?

    ϵ = schedule(U, v)

    Σ_jump = β * cov(stack(E, dims=1)) + 1e-6*I

    return (E=E, dist_E=dist_E, P=P, U=U, ϵ=ϵ, cdf_G=cdf_G, Σ_jump=Σ_jump)

end



"""
    SABC_noninf(f_dist, d_prior, r_prior, n_sample, eps_init, iter_max, v, beta, delta, resample, verbose=1, adaptjump=true, summarystats=false, y, f_summarystats; p...)

SABC.noninf Algorithm

# Arguments
- `f_dist`: Function that either returns a random sample from the likelihood or the distance between data and such a random sample. The first argument must be the parameter vector.
- `d_prior`: Function that returns the density of the prior distribution.
- `r_prior`: Function which returns one vector of a random realization of the prior distribution.
- `n_sample`: Desired number of samples.
- `eps_init`: Initial epsilon.
- `iter_max`: Maximal number of iterations.
- `v`: Tuning parameter.
- `beta`: Tuning parameter.
- `delta`: Tuning parameter.
- `resample`: After how many accepted updates?
- `p...`: Further arguments passed to f_dist, aside of the parameter.

# Returns
- A named tuple with the following keys:
    - `posterior_samples`
    - `prior_samples`
    - `eps`: Value of epsilon at final iteration.
"""
function sabc_noninf(f_dist, d_prior, r_prior, args...;
                     n_sample = 1000, resample = n_sample,
                     eps_init=1.0, iter_max=10_000,
                     v=0.3, β=1.0, δ=0.9,
                     adaptjump = true,
                     kwargs...)

    ## ------------------------
    ## Initialize
    ## ------------------------


    @unpack E, dist_E, U, ϵ, Σ_jump, cdf_G, P = initialization_noninf(f_dist, r_prior, args...;
                                                                      n_sample = n_sample,
                                                                      eps_init=eps_init,
                                                                      iter_max=iter_max,
                                                                      v=v, β=β,
                                                                      kwargs...)

    dim_par = length(first(E))


    ## --------------
    ## Sampling
    ## --------------

    progbar = ProgressMeter.Progress(iter_max, desc="Sampling...", dt=0.5)
    show_summary(ϵ, U) = () -> [(:eps, ϵ), (:mean_distance, U)]

    accept = 0
    for t in 1:iter_max

        idx = rand(1:n_sample)  # pick a particle

        # proposal
        theta_p = E[idx] + rand(Distributions.MvNormal(zeros(dim_par), Σ_jump))
        if d_prior(theta_p) > 0
            ρ_p = cdf_G(f_dist(theta_p, args...; kwargs...))

            prior_prob = d_prior(theta_p) / d_prior(E[idx])
            likeli_prob = exp((dist_E[idx] - ρ_p) / ϵ)
            accept_prob = prior_prob * likeli_prob
        else
            accept_prob = 0.0
        end

        # if isnan(accept_prob)
        #     accept_prob = ifelse((E[idx] - ρ_p) < 0, 0.0, 1.0)
        # end

        if rand() < accept_prob
            E[idx] = theta_p
            dist_E[idx] = ρ_p

            if adaptjump
                Σ_jump = β * cov(stack(E, dims=1)) + 1e-6*I
            end

            U = mean(dist_E)
            ϵ = schedule(U, v)
            accept += 1
        end


        # resample
        if (accept >= resample) && (U > eps())
            w = exp.(-dist_E .* δ ./ U)
            w = w ./ sum(w)
            idx_resampled = sample(1:n_sample, weights(w), n_sample, replace=true)
            E = E[idx_resampled, :]

            ϵ = ϵ * (1 - δ)
            U = mean(dist_E)
            ϵ = schedule(U, v)

            @info "Resampling. Effective sampling size: ", 1/sum(abs2, w ./ sum(w))
            accept = 0
        end

        # update progressbar
        ProgressMeter.next!(progbar, showvalues = show_summary(ϵ, U))
    end

    return (posterior_samples=E, prior_samples=P, eps=ϵ)
end


end
