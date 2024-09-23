# SimulatedAnnealingABC

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://eawag-siam.github.io/SimulatedAnnealingABC.jl/dev/)
[![Build Status](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl)


This package provides methods for Approximate Bayesian Computation
(ABC) (sometimes also called _simulation-based inference_ or
_likelihood-free inference_). The algorithms are based on simulated
annealing.

> [!NOTE]
> Can you evaluate the density of your posterior? Then you should most
> likely **not** be using this or any other ABC package!
> Conventional MCMC algorithm will be much more efficient.


## Installation

```Julia
] add SimulatedAnnealingABC
```

Note, Julia 1.9 or newer is needed.


## Usage

A minimal example how to use this package for approximate Bayesian inference:

```julia
using SimulatedAnnealingABC
using Distributions

# Define a stochastic model.
# Your real model should be so complex, that it would be too
# complicated to compute it's likelihood function.
function my_stochastic_model(θ, n)
    rand(Normal(θ[1], θ[2]), n)
end

# define prior of the parameters
prior = product_distribution([Normal(0,2),   # theta[1]
                              Uniform(0,2)]) # theta[2]


# Simulate some observation data
y_obs = rand(Normal(2, 0.5), 10)

# Define a function that first simulates with `my_stochastic_model` and then
# measures the distances of the simulated and the observed data with
# two summary statistics
function f_dist(θ; y_obs)
    y_sim = my_stochastic_model(θ, length(y_obs))

    (abs(mean(y_obs) - mean(y_sim)),
     abs(sum(abs2, y_obs) - sum(abs2, y_sim)) )
end


## Sample Posterior
res = sabc(f_dist, prior;
           n_particles = 1000, n_simulation = 100_000, y_obs=y_obs)


## Improve the result by running the inference for longer
res2 = update_population!(res, f_dist, prior;
                          n_simulation = 50_000, y_obs=y_obs)
```


## References

Albert, C., Künsch, H. R., & Scheidegger, A. (2015). A simulated annealing approach to approximate Bayes computations. Statistics and Computing, 25(6), 1217–1232.
