# SimulatedAnnealingABC

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://eawag-siam.github.io/SimulatedAnnealingABC.jl/dev/)
[![Build Status](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl)


This package provides methods for Approximate Bayesian Computation
(ABC) (sometimes also called _simulation-based inference_ or
_likelihood-free inference_). The algorithms are based on simulated
annealing.

> [!NOTE]
> Can you evaluate the probability density of your posterior? Can you write your
> model in `Turing.jl`? Then you should most
> likely **not** be using this or any other ABC package!
> Conventional MCMC algorithm will be much more efficient.


## Documentation

See
[here](https://eawag-siam.github.io/SimulatedAnnealingABC.jl/dev/) for
documentation and examples.


## References

Albert, C., Künsch, H. R., & Scheidegger, A. (2015). A simulated annealing approach to approximate Bayes computations. Statistics and Computing, 25(6), 1217–1232.
