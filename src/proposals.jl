## -------------------------------------------------------
##
## Define proposal generators
## -------------------------------------------------------

export RandomWalk, DifferentialEvolution, StretchMove



abstract type Proposal end

# -----------

"""
```
RandomWalk(; β=0.8, n_para)
```

Gaussian random walk proposal.

The covariance is adaptivily learned. The mixing is controlled by the tuning
parameter `β` which must be between zero and one.
"""
mutable struct RandomWalk{T} <: Proposal
    β::Float64
    Σ::T
end

function RandomWalk(; β=0.8, n_para)
    (0 < β <= 1) || error("Mixing parameter `β` must be between zero and one.")
    if n_para == 1
        RandomWalk(β, -1.0)
    else
        RandomWalk(β, -ones(n_para, n_para))
    end
end


# Random Walk proposal for n-dimensions, n > 1
function (rw::RandomWalk{T})(θ, population) where T <: AbstractArray
    log_factor = 0
    θ .+ rand(MvNormal(zeros(size(rw.Σ,1)), rw.Σ)), log_factor
end

## update covariance of the jump distributions
function update_proposal!(rw::RandomWalk{T}, population) where T <: AbstractArray
    rw.Σ .= rw.β * (cov(stack(population, dims=1)) + 1e-8*I)
end


# Random Walk proposal for 1-dimension
function (rw::RandomWalk{T})(θ, population) where T <: Real
    log_factor = 0
    θ + rand(Normal(0, sqrt(rw.Σ))), log_factor
end

## update covariance of the jump distributions
function update_proposal!(rw::RandomWalk{T}, population) where T <: Real
    rw.Σ = rw.β * cov(population)
end


# -----------

"""
```
DifferentialEvolution(; n_para, σ_gamma = 1e-5)
DifferentialEvolution(; γ0, σ_gamma = 1e-5)
```

Differential Evolution proposal, default values corresponding to EMCEE.
If the number of parameters `n_para` is provided, `γ0` is set to `2.38 / sqrt(2 * n_parameters)`.

## References

Ter Braak, C.J., 2006. A Markov Chain Monte Carlo version of the
genetic algorithm Differential Evolution: easy Bayesian computing for
real parameter spaces. Statistics and Computing 16, 239–249.

Nelson, B., Ford, E.B., Payne, M.J., 2013. Run Dmc: An Efficient,
Parallel Code For Analyzing Radial Velocity Observations Using N-Body
Integrations And Differential Evolution Markov Chain Monte Carlo. ApJS
210, 11. https://doi.org/10.1088/0067-0049/210/1/11
"""
struct DifferentialEvolution <: Proposal
    γ0::Float64
    σ_gamma::Float64

    function DifferentialEvolution(; γ0 = nothing, n_para = nothing, σ_gamma = 1e-5)
        if !isnothing(γ0) && isnothing(n_para)
            new(γ0, σ_gamma)
        elseif !isnothing(n_para) && isnothing(γ0)
            γ0 = 2.38/sqrt(2*n_para)
            new(γ0, σ_gamma)
        else
            throw(ArgumentError("Provide either `γ0` or `n_para`, not both."))
        end
    end
end

function (de::DifferentialEvolution)(θ, population)
    # sample index of two different partner particles
    i1 = i2 = 0
    while i1 == i2
        i1 = rand(1:length(population))
        i2 = rand(1:length(population))
    end

    # propose move
    γ = de.γ0 * (1 + de.σ_gamma * randn()) # based on Nelson et al, 2013

    log_factor = 0
    θ .+ γ .* (population[i1] .- population[i2]), log_factor
end

update_proposal!(de::DifferentialEvolution, population) = nothing



# -----------

"""
```
StretchMove(;a=2)
```
The standard proposal used in EMCEE.

## Reference

Goodman, J., Weare, J., 2010. Ensemble samplers with affine invariance. Communications in Applied Mathematics and Computational Science 5, 65–80.
"""
struct StretchMove <: Proposal
    a::Float64
end
StretchMove(;a=2) = StretchMove(a)

function (sm::StretchMove)(θ, population)
    # sample index of a partner particle
    # Note, because we split the population batches,
    # it is ensured to be different from θ.
    i = rand(1:length(population))

    # proposed move
    z = ((sm.a - 1.0) * rand() + 1)^2 / sm.a

    log_factor = log(z) * (length(θ) - 1)
    population[i] .+ z .* (θ .- population[i]), log_factor
end

update_proposal!(sm::StretchMove, population) = nothing
