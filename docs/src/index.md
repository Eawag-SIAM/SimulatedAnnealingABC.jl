```@meta
CurrentModule = SimulatedAnnealingABC
```

# SimulatedAnnealingABC

Documentation for [SimulatedAnnealingABC.jl](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl).

This package provides different SimulatedAnnealingABC (SABC)
algorithms for Approximate Bayesian Computation (ABC). Other terms
that are sometimes used for ABC are _simulation-based inference_ or
_likelihood-free inference_.
:

ABC is well-suited for models where evaluating the likelihood function
``p(D \mid θ)`` is computationally expensive, but sampling from the
likelihood is relatively easy. This is often true for stochastic
models with unobserved random states ``z``:

``p(D \mid θ) = \int p(D \mid z, θ) p(z) \, \text{d}z``

If ``z`` is high-dimensional, the integration may become so computational
expensive that conventional MCMC algorithms are no longer feasible.


!!! note

    Can you evaluate the density of your posterior? Can you write your
    model in `Turing.jl`? Then you should
    most likely **not** be using this or any other ABC package!
    Conventional MCMC algorithms will be much more efficient.





## References

Albert, C., Künsch, H.R., Scheidegger, A., 2015. A simulated annealing
approach to approximate Bayes computations. Statistics and computing
25, 1217–1232. [https://doi.org/10.1007/s11222-014-9507-8](https://doi.org/10.1007/s11222-014-9507-8)
