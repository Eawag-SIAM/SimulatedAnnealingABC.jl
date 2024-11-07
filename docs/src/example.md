# Example

We use a simple stochastic SIR model to demonstrate the usage of
`SimulatedAnnealingABC.jl` for Bayesian inference.

## Stochastic SIR Model

The stochastic SIR (Susceptible-Infected-Recovered) model describes
the spread of an infectious disease in a closed population of ``N``
individuals. Transmission and recovery events happen at random
times. The model has two parameters:

- ``\beta``: Infection rate per contact per unit time.
- ``\gamma``: Recovery rate per infected individual per unit time.

The number of susceptible, infected, and recovered individuals at time
``t``, are denoted by ``S(t)``, ``I(t)``, and ``R(t)``.

The infection rate at a given time depends on the number of susceptible and infected individuals

``\lambda_{\text{infection}}(t) = \frac{\beta S(t) I(t)}{N}``,

while the recovery rate

``\lambda_{\text{recovery}})(t) = \gamma I(t)``

depends only on the infected individual.

Assuming a Poisson process, the time until the next event (infection or recovery), ``\Delta t``, is  exponential
distributed:
``\Delta t \sim \text{Exponential}(\lambda_{\text{total}})``

### Inference

#### Observations

We cannot observe every individual. Instead, let's assume that we
observe three key figures:

1. the total number infected individuals,
2. the number of infected individuals at the peak of the wave, and
3. the time point when the wave peaked.

Note, that it would be very difficult to derive the density of the likelihood
function ``p(\text{observations} \mid \gamma, \beta)`` for the stochastic SIR model for this kind of
observations. At the same time it is not difficult to simulate such
data from our model. Therefore ABC algorithms are a good fit for this inference problem.

#### Prior

For Bayesian inference we need to define a prior distribution for
the parameters. Let's assume the prior for the parameters are
independent uniform distributions:
- Transmission rate: ``\beta \sim \text{Uniform}(0.1, 1)``
- Recovered rate: ``\gamma \sim \text{Uniform}(0.05, 0.5)``




### Implementation

First we load the required packages and set a random seed.
```@example 1
using SimulatedAnnealingABC
using Distributions
using Plots

import Random
Random.seed!(123)
nothing # hide
```

Then we define the stochastic SIR model
```@example 1
function stochastic_SIR(θ; S0::Int, I0::Int, R0::Int, t_max)

    β, γ = θ
    # Initialize variables
    S = S0
    I = I0
    R = R0
    t = 0.0
    N = S0 + I0 + R0  # Total population

    # Records of the simulation
    time = [t]
    S_record = [S]
    I_record = [I]
    R_record = [R]

    while t < t_max && I > 0
        # Calculate rates
        infection_rate = β * S * I / N
        recovery_rate = γ * I
        total_rate = infection_rate + recovery_rate

        # Time to next event
        dt = rand(Exponential(1 / total_rate))
        t += dt

        # Determine which event occurs
        if rand() < infection_rate / total_rate
            # Infection event
            S -= 1
            I += 1
        else
            # Recovery event
            I -= 1
            R += 1
        end

        # Record the state
        push!(time, t)
        push!(S_record, S)
        push!(I_record, I)
        push!(R_record, R)
    end


    return (time = time, S = S_record,
            I = I_record, R = R_record)
end
nothing  # hide
```

For the sake of this example we simulate data. Note, the random seed has a large influence on how the simulation
result looks like.
```@example 1
# "true" parameters
θtrue = [0.3,    # Transmission rate β
         0.1]    # Recovery rate γ

sim = stochastic_SIR(θtrue; S0=99, I0=1, R0=0, t_max=160)

# Plot the results
plot(sim.time, sim.S, label="Susceptible", xlabel="Time (days)", ylabel="Number of Individuals",
     title="Stochastic SIR Model", linewidth=2, linetype=:steppre)
plot!(sim.time, sim.I, label="Infected", linewidth=2, linetype=:steppre)
plot!(sim.time, sim.R, label="Recovered", linewidth=2, linetype=:steppre)
```
The observations are defined as follows:
```@example 1
data_obs = (
    total_infected = sim.R[end],
    peak_infected = maximum(sim.I),
    t_peak = sim.time[argmax(sim.I)]
)
```

To use `sabc` we need to define a function that returns one or more
distances between a random simulation from out model with parameter
``\theta`` and the observed data. Here we define two functions for demonstration. The
first one returns the distance of each observed statistics separately and the
second one aggregates them all into a single distance.
```@example 1
function f_dist_multi_stats(θ, data_obs)
    # run model
    sim = stochastic_SIR(θ; S0=99, I0=1, R0=0, t_max=160)

    sim_total_infected = sim.R[end]
    sim_peak_infected = maximum(sim.I)
    sim_t_peak = sim.time[argmax(sim.I)]

    # compute distance of summary statistics
    dists = (abs2(sim_total_infected - data_obs.total_infected),
             abs2(sim_peak_infected - data_obs.peak_infected),
             abs2(sim_t_peak - data_obs.t_peak))

end

# For single stat version we need to aggregate the statistics.
# Here we give the same weight to each statistic
f_dist_single_stat(θ, data_obs) = sum(f_dist_multi_stats(θ, data_obs))
nothing # hide
```

It is often not clear, how to aggregating the distances of multiple
statistics meaningfully. For example, here we had to combined distances
measured in different units (number of individuals and time in days).

The prior is defined with `Distributions.jl`:
```@example 1
prior = product_distribution([Uniform(0.1, 1),     # β
                              Uniform(0.05, 0.5)]) # γ
```

With everything in place, we run the inference for both distances. Note, that the argument
`data_obs` is passed to the distance functions.
```@example 1
res_1 = sabc(f_dist_single_stat, prior, data_obs;
             n_simulation = 500_000,
             n_particles = 5000)
```

```@example 1
res_2 = sabc(f_dist_multi_stats, prior, data_obs;
             n_simulation = 500_000,
             n_particles = 5000)
```

Finally, we plot the posterior parameter distribution. The multiple
statistics are a bit more informative resulting in a narrower posterior.
```@example 1
pop_1 = stack(res_1.population)
pop_2 = stack(res_2.population)

p1 = scatter(pop_1[1,:], pop_1[2,:], title="single stat",
             markeralpha = 0.3,
             markersize = 1,
             markerstrokewidth = 0);
scatter!(p1, θtrue[1:1], θtrue[2:2], markersize = 2);

p2 = scatter(pop_2[1,:], pop_2[2,:], title="multiple stats",
             markeralpha = 0.3,
             markersize = 1,
             markerstrokewidth = 0);
scatter!(p2, θtrue[1:1], θtrue[2:2], markersize = 2);

plot(p1, p2, xlab = "β", ylab = "γ", legend=false)
```


We can try different strategies to generate proposal jumps in the
parameter space:

```@example 1
res_1 = sabc(f_dist_multi_stats, prior, data_obs;
             n_simulation = 200_000,
             n_particles = 10_000,
             proposal = DifferentialEvolution(γ0=2.38/sqrt(2*2)),
             show_progressbar=true)

res_2 = sabc(f_dist_multi_stats, prior, data_obs;
             n_simulation = 200_000,
             n_particles = 10_000,
             proposal = StretchMove(),
             show_progressbar=true)

res_3 = sabc(f_dist_multi_stats, prior, data_obs;
             n_simulation = 200_000,
             n_particles = 10_000,
             proposal = RandomWalk(β=0.8, n_para=2),
             show_progressbar=true)
nothing
```

For this simple two-dimensional example the results are about the
same:

```@example 1
pop_1 = stack(res_1.population)
pop_2 = stack(res_2.population)
pop_3 = stack(res_3.population)

p1 = scatter(pop_1[1,:], pop_1[2,:], title="Differential Evolution",
             markeralpha = 0.2,
             markersize = 1,
             markerstrokewidth = 0);
scatter!(p1, θtrue[1:1], θtrue[2:2], markersize = 2);

p2 = scatter(pop_2[1,:], pop_2[2,:], title="Stretch Move",
             markeralpha = 0.2,
             markersize = 1,
             markerstrokewidth = 0);
scatter!(p2, θtrue[1:1], θtrue[2:2], markersize = 2);

p3 = scatter(pop_3[1,:], pop_3[2,:], title="Gaussian random walk",
             markeralpha = 0.2,
             markersize = 1,
             markerstrokewidth = 0);
scatter!(p3, θtrue[1:1], θtrue[2:2], markersize = 2);

plot(
    p1, p2, p3,
    xlab = "β", ylab = "γ", legend=false,

)

```
