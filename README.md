# SimulatedAnnealingABC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://scheidan.github.io/SimulatedAnnealingABC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://scheidan.github.io/SimulatedAnnealingABC.jl/dev/)
[![Build Status](https://github.com/scheidan/SimulatedAnnealingABC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/SimulatedAnnealingABC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/scheidan/SimulatedAnnealingABC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/SimulatedAnnealingABC.jl)


## Installation

```Julia
] add https://github.com/scheidan/SimulatedAnnealingABC.jl
```
Note, Julia 1.9 or newer is needed.

## Usage

```Julia
using SimulatedAnnealingABC
using Distributions
import MCMCChains (optional)

## Define model
f_dist(θ) = sum(abs2, rand(Normal(θ[1], θ[2]), 4))

# define prior
prior = product_distribution([Normal(0,1),   # theta[1]
                              Uniform(0,1)]) # theta[2]

## Sample Posterior
res = sabc(f_dist, prior; eps_init = 1,
           n_particles = 100, n_simulation = 10_000)

## MCMCChains can help to work with the samples
chn = MCMCChains.Chains(res.posterior_samples)
```

## References

Albert, C., Künsch, H. R., & Scheidegger, A. (2015). A simulated annealing approach to approximate Bayes computations. Statistics and Computing, 25(6), 1217–1232.
