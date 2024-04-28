# SimulatedAnnealingABC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://scheidan.github.io/SimulatedAnnealingABC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://scheidan.github.io/SimulatedAnnealingABC.jl/dev/)
[![Build Status](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Eawag-SIAM/SimulatedAnnealingABC.jl)


## Installation

```Julia
] add https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl
```
Note, Julia 1.9 or newer is needed.

## Usage

```Julia
using SimulatedAnnealingABC
using Distributions
using Random
using Distances

## Generate data
# ----------------- #
Random.seed!()
true_mean = 3; true_sigma = 15
num_samples = 100
yobs = rand(Normal(true_mean, true_sigma), num_samples)
# ----------------- #

## Summary stats for obsrevations
# ----------------- #
s1obs = mean(yobs); s2obs = std(yobs) 
ss_obs = [s1obs, s2obs]
# ----------------- #

## Define prior
# ----------------- #
s1_min = -10; s1_max = 20    # stat 1 (distribution mean)
s2_min = 0; s2_max = 25      # stat 2 (distribution std)
prior = product_distribution(Uniform(s1_min, s1_max), Uniform(s2_min, s2_max))
# ----------------- #

## Model + distance function
## N.B.: DISTANCE MUST BE A VECTOR CONTAINING INDIVIDUAL DISTANCES FOR EACH STAT
# ----------------- #
function f_dist(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]), num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y)
	ss = [s1, s2]
	# DistancE
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end
# ----------------- #

## Sample Posterior
# ----------------- #
nsim = 2_000_000  # total number of particle updates
# --- TYPE 1 -> single-epsilon ---
out_singeps = sabc(f_dist, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, type = 1)
display(out_singeps)
# --- TYPE 2 -> multi-epsilon ---
out_multeps = sabc(f_dist, prior; n_particles = 1000, v = 10.0, n_simulation = nsim, type = 2)
display(out_multeps)
# --- TYPE 3 -> hybrid multi-u-single-epsilon ---
out_hybrid = sabc(f_dist, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, type = 3)
display(out_hybrid)
# ----------------- #

## Extract posterior population, trajectories for epsilon, rho and u
# ----------------- #
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

pop_hybrid = hcat(out_hybrid.population...)
eps_hybrid = hcat(out_hybrid.state.ϵ_history...)
rho_hybrid = hcat(out_hybrid.state.ρ_history...)
u_hybrid = hcat(out_hybrid.state.u_history...)

## Update existing population with another 1_000_000 simulations
# ----------------- #
v = 1.0   # v = 10 for multi-epsilon
# replace 'out' with out_singeps, out_multeps or out_hybrid
# choose the corresponding type for the algorithm
type = 1,2 or 3
update_population!(out, f_dist, prior; v = v, n_simulation = 10_000, type = type)

```

## References

Albert, C., Künsch, H. R., & Scheidegger, A. (2015). A simulated annealing approach to approximate Bayes computations. Statistics and Computing, 25(6), 1217–1232.
