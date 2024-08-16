```@meta
CurrentModule = SimulatedAnnealingABC
```

# SimulatedAnnealingABC

Documentation for [SimulatedAnnealingABC](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl).

This package provides different SimulatedAnnealingABC (SABC)
algorithms for Approximate Bayesian Computation (ABC) (sometimes also called
_simulation-based inference_).

> [!NOTE]
> If you are able to compute the density of your posterior, you should
most likely not be using this or another ABC package.  A traditional MCMC
algorithm will be much more efficient.



## Usage

todo...



## API

```@index
```

```@docs
sabc
```

```@docs
 update_population!
```


## Related Julia Packages

todo

## References

Albert, C., Künsch, H.R., Scheidegger, A., 2015. A simulated annealing
approach to approximate Bayes computations. Statistics and computing
25, 1217–1232. https://doi.org/10.1007/s11222-014-9507-8
