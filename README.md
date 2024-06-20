# SimulatedAnnealingABC

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://scheidan.github.io/SimulatedAnnealingABC.jl/dev/)
[![Build Status](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl)

## Installation

The package is under development and not yet officially registered. You can install it from GitHub with

```Julia
] add https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl
```

Note, Julia 1.9 or newer is needed.

## Usage

Here we show how to run the SimulatedAnnealingABC (SABC) algorithm to infer model parameters given some observed dataset.

### Data (observations)

Let us start off by generating a synthetic dataset, our "observations". Consider for instance a normally distributed random sample.  

```Julia
using Random
using Distributions

Random.seed!(1111)					  
true_μ = 3                                                # "true" μ - to be inferred
true_σ = 15                                               # "true" σ - to be inferred
num_samples = 100                                         # dataset size
y_obs = rand(Normal(true_μ, true_σ), num_samples)         # our "observations"
```

### Model

Now, we need to define a **data-generating model**. In this example, quite obviously, we opt for a Gaussian model $\mathcal{N}(\mu,\sigma)$. The goal is to infer a posterior distribution for the parameters $\mu$ and $\sigma$ given the dataset `y_obs`.

```Julia
# Data-generating model
# θ = [μ,σ] - parameter set
function model(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return y
end
```

For a given parameter set $\theta = \left( \mu, \sigma \right)$, the model function generates a dataset of size `num_samples`.

### Prior for model parameters

We also need to define a **prior** distribution for the model parameters that are to be inferred. In this case we choose a `Uniform` distribution for both parameters.

```Julia
# Prior for the parameters to be inferred 
μ_min = -10; μ_max = 20    # parameter 1: mean
σ_min = 0; σ_max = 25      # parameter 2: std
prior = product_distribution(Uniform(μ_min, μ_max), Uniform(σ_min, σ_max))
```

### Summary statistics

In general, the core of ABC algorithms consists in using the model as a forward simulator to generate a large number of synthetic datasets, for different sets of model parameters. The latter are then accepted or rejected by measuring the discrepancy (or "distance", according to some metric) between the observations and the model-generated data. However, an inference based on the direct comparison of observations and model-based data is not always feasible, e.g., in case of high dimensional datasets. In such cases, the discrepancy is measured by computing a distance between relevant summary statistics extracted from the datasets. In our toy example, we use the empirical mean and standard deviation of the data as **summary statistics**.

```Julia
function sum_stats(data)
	stat1 = mean(data)
	stat2 = std(data) 
	return [stat1, stat2]
end
```

The function `sum_stats` returns an array of size `n_stats`, where `n_stats` denotes the number of summary statistics (2 in our case).

```Julia
n_stats = size(sum_stats(y_obs), 1)
```

We can now reduce our "observations" to a (low-dimensional) set of summary statistics. 

```Julia
ss_obs = sum_stats(y_obs)
```

### Re-definition of the model

The inference algorithm will compare observations and model outputs in terms of summary statistics. Therefore, we can reformulate the definition of the data-generating model in such a way that the data are directly compressed into a corresponding set of summary statistics.

```Julia
# Data-generating model + reduction to summary statistics
# θ = [μ,σ] - parameter set
function model(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return sum_stats(y)
end
```

Now, the `model` function returns a low-dimensional array of length `n_stats`.

### Distance function

Lastly, we need to define a **distance function**. The distance function MUST return an array containing the INDIVIDUAL distances, for EACH summary statistics, between observations and model outputs. In other words, given `n_stats` summary statistics, the distance function will ALWAYS return an array of length `n_stats`.  

```Julia
using Distances

function f_dist(θ)
	ss = model(θ)    # data-generating model (returns summary statistics)
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:n_stats]   # distance
	return rho
end
```

In this way, the algorithm will keep track of the evolution of EACH individual distance (for EACH summary statistics), although the SABC algorithm effectively uses only one distance given by the square root of the sum of the squared individual distances:

```Julia
sqrt(sum(f_dist(θ).^2))
```

### SABC inference

The following functions have been implemented up to this point:

1. `sum_stats` (reduces a high-dimensional dataset to a low-dimensional set of statistics) 
2. `model` (given model parameters `θ`, generates a dataset and returns summary statistics)
3. `f_dist` (computes the distance between individual summary statistics)
4. `prior` (prior for the parameters to be inferred)

We are now ready to run the SABC inference!

We decide to use 1000 particles and start with 1 million particle updates (the algorithm allows us to continue the inference afterwards).

```Julia
np = 1000       # number of particles
ns = 1_000_000  # number of particle updates
```

Note that the total number of **population updates** is given by `n_pop_updates = div(ns, np)`. Note also that the generation of the initial prior sample is counted as a population update step. Therefore, the total number of population updates after initialization will effectively be `n_pop_updates - 1`.  

The `sabc` function requires two inputs:

1. the distance function `f_dist`, which contains the data generating `model`,
2. and the `prior`.

Additional arguments that can be passed to the function are the following.

- `n_particles`: number of particles used in the inference.
- `n_simulation`: total number of particle updates.
- `v`: annealing speed. Recommended value is `v=1`.
- `checkpoint_display`: every how many population updates information on the inference progress is displayed. Default value is `100`. Recommended value with a long inference is `500` or more (to avoid unnecessarily lengthy output files). Note that `n` population updates correspond to `n*n_particles` particle updates.    

The following arguments rarely require adjustments.

- `resample`: number of successful particle updates between consecutive particle resampling events. Default value is `2*n_particles`.
- `β`: tuning hyperparameter governing the equilibration/mixing speed, used to rescale the jump covariance matrix. Default value is `β=0.8`.
- `δ`: tuning hyperparameter governing the size of population resampling. Default value is `δ=0.1`.
- `checkpoint_history`: every how many population updates information on the inference progress is stored for postprocessing. Default value is `1`.

Let us run the inference:

```Julia
using SimulatedAnnealingABC

out_single_eps = sabc(f_dist, prior; n_particles = np, n_simulation = ns, v = 1.0)
display(out_single_eps)

```

### Get SABC output

We can now extract posterior population as well as trajectories for ϵ's, ρ's (distances) and u's ("transformed" distances).

```Julia
pop_singeps = hcat(out_single_eps.population...)
eps_singeps = hcat(out_single_eps.state.ϵ_history...)
rho_singeps = hcat(out_single_eps.state.ρ_history...)
u_singeps = hcat(out_single_eps.state.u_history...)

```

The output files have the following dimensions:

- **populations**: `(n_stats, np)`
- **ϵ trajectories**: `(1, n_pop_updates)`
- **ρ trajectories**: `(n_stats, n_pop_updates + 1)`
- **u trajectories**: `(1, n_pop_updates)`

**Remarks about ρ trajectories**:

- ρ trajectories are always stored as individual trajectories for each summary statistics, even if a single ρ is effectively used for the inference.
- Moreover, note that each trajectory has length `n_pop_updates + 1`. This is because also the distances of the prior sample, **before rescaling**, are stored as first element of the ρ array.

### To continue the inference 

After analysing the output, we may decide to update the current population with another 1_000_000 simulations. That can be easily done with the function `update_population!`.

```Julia
out_single_eps_2 = update_population!(out_single_eps, f_dist, prior; n_simulation = ns, v = 1.0)
```

One may also want to store the output file and decide whether to continue the inference later. One way to save SABC output files is to use `serialize`. Here is an example.

```Julia
using Serialization

# replace [output path] with appropriate path
serialize("[output path]/sabc_result_single_eps", out_single_eps) 

```

To continue the inference one can proceed as follows.

```Julia
# replace [output path] with appropriate path
out_single_eps_1 = deserialize("[output path]/sabc_result_single_eps")
out_single_eps_2 = update_population!(out_single_eps_1, f_dist, prior; n_simulation = 1_000_000, v = 1.0)
```

### Results

A sample from the true posterior can be easily obtained using the [AffineInvariantMCMC](https://github.com/madsjulia/AffineInvariantMCMC.jl) package as follows.

```Julia
using AffineInvariantMCMC

# log-likelihood 
llhood = θ -> begin
	μ, σ  = θ;
	return -num_samples*log(σ) - sum((y_obs.-μ).^2)/(2*σ^2)
end

# log-prior
lprior = θ -> begin
	μ, σ  = θ;
	if (μ_min <= μ <= μ_max) && (σ_min <= σ <= σ_max)
		return 0.0
	else
		return -Inf
	end
end

# log-posterior
lprob = θ -> begin
	μ, σ  = θ;
	lp = lprior(theta)
	if isinf(lp) 
		return -Inf
	else
		return lp + llhood(theta)
	end
end

# Generate 1000 posterior samples
numdims = 2
numwalkers = 10
thinning = 10
numsamples_perwalker = 1000
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(μ_min, μ_max), numwalkers); 
theta0[2, :] = rand(rng, Uniform(σ_min, σ_max), numwalkers); 

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

# plot
scatter(flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true posterior", xlims = (σ_min,σ_max), ylims = (μ_min,μ_max), xlabel = "std", ylabel = "mean")

```

<img width="497" alt="image" src="https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/assets/9136326/8c01d837-6144-457c-95a1-0629ac8c4e17">

The plot area corresponds to the prior range. Here is the posterior sample after 2 million particle updates obtained with the SABC algorithm, compared to the true posterior.

<img width="497" alt="image" src="https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/assets/9136326/3083a9ee-eac1-43d9-a99e-1fefa8168b03">


## References

Albert, C., Künsch, H. R., & Scheidegger, A. (2015). A simulated annealing approach to approximate Bayes computations. Statistics and Computing, 25(6), 1217–1232.
