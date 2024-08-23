```@meta
CurrentModule = SimulatedAnnealingABC
```

# SimulatedAnnealingABC

Documentation for [SimulatedAnnealingABC](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl).

This package provides different SimulatedAnnealingABC (SABC)
algorithms for Approximate Bayesian Computation (ABC) (sometimes also called
_simulation-based inference_ or _likelihood-free inference_).

!!! note

    Can you evaluate the density of your posterior? Then you should
    most likely **not** be using this or any other ABC package!  A traditional MCMC
    algorithm will be much more efficient.



## Usage

### Getting started

A minimal example how to use this package for inference:

```julia
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

# Define a function that simulate with my_stochastic_model and then
# measure the distances of the simulated and the observed data with
# two summary statistics
function f_dist(θ; y_obs)
    y_sim = my_stochastic_model(θ, length(y_obs))

    (
        abs(mean(y_obs) - mean(y_sim)),
        abs(sum(abs2, y_obs) - sum(abs2, y_sim))
    )
end


## Sample Posterior
res = sabc(f_dist, prior;
           n_particles = 1000, n_simulation = 100_000, y_obs=y_obs)


## Improve the results by running the inference for longer
res2 = update_population!(res, f_dist, prior;
                          n_simulation = 50_000, y_obs=y_obs)

```

### Logging and progress bar

There are two ways `sabc` can inform about the progress of the ongoing inference run:

- Progress bar. By default it is shown in interactive sessions. It can be disabled with the argument
  (`show_progressbar = false`)

- Logging statements. The argument `show_checkpoint` controls how
  often a summary of the current state is logged. For long running
  computations in a cluster environment this is more convenient
  than the progress bar.

You can set the logging level to `Warn` to suppress all logging
statements:

```Julia
using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

... run sabc() ...
```

### How to disable multi-threading

- TODO


## Worked example

Translate and update example from readme.

explain pro and cons of the different algo types.


## API

```@docs
sabc
```

```@docs
update_population!
```

```@docs
SABCresult
```

## Related Julia Packages

### Approximate Bayesian Computation

- [`ABCdeZ.jl`](https://github.com/mauricelanghinrichs/ABCdeZ.jl) -
  Approximate Bayesian Computation (ABC) with differential evolution
  (de) moves and model evidence (Z) estimates.

- [`ApproxBayes.jl`](https://github.com/marcjwilliams1/ApproxBayes.jl) -
  Implements basic ABC rejection sampler and sequential monte carlo
  algorithm (ABC SMC) as in Toni. et al 2009 as well as model
  selection versions of both (Toni. et al 2010).

- [`GpABC.jl`](https://github.com/tanhevg/GpABC.jl) - ABC with
  emulation on Gaussian Process Regression.

- [`KissABC.jl`](https://github.com/francescoalemanno/KissABC.jl) -
  Implementation of Multiple Affine Invariant Sampling for efficient
  Approximate Bayesian Computation. Is looking for a new maintainer.


### Misc

- [`SimulationBasedInference.jl`](https://github.com/bgroenks96/SimulationBasedInference.jl)
  Despite the name this package seems to focus on traditional Bayesian
  inference methods and not ABC, i.e. it assumes we can evaluate the density of
  the posterior.


## References

Albert, C., Künsch, H.R., Scheidegger, A., 2015. A simulated annealing
approach to approximate Bayes computations. Statistics and computing
25, 1217–1232. [https://doi.org/10.1007/s11222-014-9507-8](https://doi.org/10.1007/s11222-014-9507-8)
