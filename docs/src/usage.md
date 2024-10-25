# Usage

## Getting started

This is a minimal example using this package for inference. See
the example section for more in-depth explanations.

```@example
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

## Logging and progress bar

There are two ways `sabc` can inform about the progress of the ongoing inference run:

- **Progress bar** By default it is shown in interactive sessions. It can be disabled with the argument
  (`show_progressbar = false`)

- **Logging statements** The argument `show_checkpoint` controls how
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
