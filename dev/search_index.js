var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"sabc","category":"page"},{"location":"api/#SimulatedAnnealingABC.sabc","page":"API","title":"SimulatedAnnealingABC.sabc","text":"sabc(f_dist::Function, prior::Distribution, args...;\n      n_particles = 100, n_simulation = 10_000,\n      algorithm = :single_eps,\n      resample = 2*n_particles,\n      v=1.0, β=0.8, δ=0.1,\n      checkpoint_history = 1,\n      show_progressbar::Bool = !is_logging(stderr),\n      show_checkpoint = is_logging(stderr) ? 100 : Inf,\n      kwargs...)\n\nSimulated Annealing Approximate Bayesian Inference Algorithm\n\nArguments\n\nf_dist: Function that returns one or more distances between data and a random sample from the likelihood. The first argument must be the parameter vector.\nprior: A Distribution defining the prior.\nargs...: Further arguments passed to f_dist\nn_particles: Desired number of particles.\nn_simulation: maximal number of simulations from f_dist.\nv = 1.0: Tuning parameter for annealing speed. Must be positive.\nβ = 0.8: Tuning parameter for mixing. Between zero and one.\nδ = 0.1: Tuning parameter for resampling intensity. Must be positive and should be small.\nalgorithm = :single_eps: Choose algorithm, either :multi_eps, or :single_eps. With :single_eps a global tolerance is used for all distances. Wit :multi_eps every distnace has it's own tolerance.\nresample: After how many accepted population updates?\ncheckpoint_history = 1: every how many population updates distances and epsilons are stored\nshow_progressbar::Bool = !is_logging(stderr): defaults to true for interactive use.\nshow_checkpoint::Int = 100: every how many population updates algorithm state is displayed.                               By default disabled for for interactive use.\nkwargs...: Further arguments passed to f_dist`\n\nReturn\n\nAn object of type SABCresult\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"update_population!","category":"page"},{"location":"api/#SimulatedAnnealingABC.update_population!","page":"API","title":"SimulatedAnnealingABC.update_population!","text":"update_population!(population_state::SABCresult,\n                   f_dist, prior, args...;\n                   n_simulation,\n                   v=1.0, β=0.8, δ=0.1,\n                   resample = 2*length(population_state.population),\n                   checkpoint_history = 1,\n                   show_progressbar::Bool = !is_logging(stderr),\n                   show_checkpoint = is_logging(stderr) ? 100 : Inf,\n                   kwargs...)\n\nUpdates particles with n_simulation and applies importance sampling if needed. Modifies population_state.\n\nArguments\n\nSee docstring for sabc.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"SimulatedAnnealingABC.SABCresult","category":"page"},{"location":"api/#SimulatedAnnealingABC.SABCresult","page":"API","title":"SimulatedAnnealingABC.SABCresult","text":"Holds results from a SABC run with fields:\n\npopulation: vector of parameter samples from the approximate posterior\nu: transformed distances\nρ: distances\nstate: state of algorithm\n\nThe history of ϵ can be accessed with the field state.ϵ_history. The history of ρ can be accessed with the field state.ρ_history. The history of u can be accessed with the field state.u_history.\n\n\n\n\n\n","category":"type"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/#Getting-started","page":"Usage","title":"Getting started","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"A minimal example how to use this package for inference:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using SimulatedAnnealingABC\nusing Distributions\n\n# Define a stochastic model.\n# Your real model should be so complex, that it would be too\n# complicated to compute it's likelihood function.\nfunction my_stochastic_model(θ, n)\n    rand(Normal(θ[1], θ[2]), n)\nend\n\n# define prior of the parameters\nprior = product_distribution([Normal(0,2),   # theta[1]\n                              Uniform(0,2)]) # theta[2]\n\n\n# Simulate some observation data\ny_obs = rand(Normal(2, 0.5), 10)\n\n# Define a function that first simulates with `my_stochastic_model` and then\n# measures the distances of the simulated and the observed data with\n# two summary statistics\nfunction f_dist(θ; y_obs)\n    y_sim = my_stochastic_model(θ, length(y_obs))\n\n    (abs(mean(y_obs) - mean(y_sim)),\n     abs(sum(abs2, y_obs) - sum(abs2, y_sim)) )\nend\n\n\n## Sample Posterior\nres = sabc(f_dist, prior;\n           n_particles = 1000, n_simulation = 100_000, y_obs=y_obs)\n\n\n## Improve the result by running the inference for longer\nres2 = update_population!(res, f_dist, prior;\n                          n_simulation = 50_000, y_obs=y_obs)","category":"page"},{"location":"usage/#Logging-and-progress-bar","page":"Usage","title":"Logging and progress bar","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"There are two ways sabc can inform about the progress of the ongoing inference run:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Progress bar By default it is shown in interactive sessions. It can be disabled with the argument (show_progressbar = false)\nLogging statements The argument show_checkpoint controls how often a summary of the current state is logged. For long running computations in a cluster environment this is more convenient than the progress bar.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"You can set the logging level to Warn to suppress all logging statements:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using Logging\nglobal_logger(ConsoleLogger(stderr, Logging.Warn))\n\n... run sabc() ...","category":"page"},{"location":"related/#Related-Packages","page":"Related packages","title":"Related Packages","text":"","category":"section"},{"location":"related/#Julia","page":"Related packages","title":"Julia","text":"","category":"section"},{"location":"related/#Approximate-Bayesian-Computation","page":"Related packages","title":"Approximate Bayesian Computation","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"ABCdeZ.jl - Approximate Bayesian Computation (ABC) with differential evolution (de) moves and model evidence (Z) estimates.\nApproxBayes.jl - Implements basic ABC rejection sampler and sequential monte carlo algorithm (ABC SMC) as in Toni. et al 2009 as well as model selection versions of both (Toni. et al 2010).\nGpABC.jl - ABC with emulation on Gaussian Process Regression.\nKissABC.jl - Implementation of Multiple Affine Invariant Sampling for efficient Approximate Bayesian Computation. Is looking for a new maintainer.","category":"page"},{"location":"related/#Misc","page":"Related packages","title":"Misc","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"SimulationBasedInference.jl Despite the name this package seems to focus on traditional Bayesian inference methods and not ABC, i.e. it assumes we can evaluate the density of the posterior.","category":"page"},{"location":"related/#R","page":"Related packages","title":"R","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"abc","category":"page"},{"location":"related/#Python","page":"Related packages","title":"Python","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"pyABC","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"CurrentModule = SimulatedAnnealingABC","category":"page"},{"location":"#SimulatedAnnealingABC","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"","category":"section"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"Documentation for SimulatedAnnealingABC.jl.","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"This package provides different SimulatedAnnealingABC (SABC) algorithms for Approximate Bayesian Computation (ABC). Other terms that are sometimes used for ABC are simulation-based inference or likelihood-free inference. :","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"ABC is well-suited for models where evaluating the likelihood function p(D mid θ) is computationally expensive, but sampling from the likelihood is relatively easy. This is often true for stochastic models with unobserved random states z:","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"p(D mid θ) = int p(D mid z θ) p(z)  textdz","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"If z is high-dimensional, the integration may become so computational expensive that conventional MCMC algorithms are no longer feasible.","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"note: Note\nCan you evaluate the density of your posterior? Can you write your model in Turing.jl? Then you should most likely not be using this or any other ABC package! Conventional MCMC algorithms will be much more efficient.","category":"page"},{"location":"#References","page":"SimulatedAnnealingABC","title":"References","text":"","category":"section"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"Albert, C., Künsch, H.R., Scheidegger, A., 2015. A simulated annealing approach to approximate Bayes computations. Statistics and computing 25, 1217–1232. https://doi.org/10.1007/s11222-014-9507-8","category":"page"},{"location":"example/#Example","page":"Example","title":"Example","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We use a stochastic SIR model to demonstrate the usage of SimulatedAnnealingABC.jl.","category":"page"},{"location":"example/#Stochastic-SIR-Model","page":"Example","title":"Stochastic SIR Model","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"A stochastic SIR (Susceptible-Infected-Recovered) model describes the spread of an infectious disease in a closed population of N individuals. Transmission and recovery events happen at random times. The model parameters are:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"beta: Infection rate per contact per unit time.\ngamma: Recovery rate per infected individual per unit time.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The number of susceptible, infected, and recovered individuals at time t, are denoted by S(t), I(t), and R(t).","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The infection rate at a given time depends on the number of susceptible and infected individuals","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"lambda_textinfection(t) = fracbeta S(t) I(t)N,","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"while the recovery rate","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"lambda_textrecovery)(t) = gamma I(t)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"depends only on the infected individual.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Assuming a Poisson process, the time until the next event (infection or recovery), Delta t, is  exponential distributed: Delta t sim textExponential(lambda_texttotal)","category":"page"},{"location":"example/#Inference","page":"Example","title":"Inference","text":"","category":"section"},{"location":"example/#Observations","page":"Example","title":"Observations","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We cannot observe every individual. Instead, let's assume that we observe three key figures:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"the total number infected,\nthe number of infected at the peak of the wave, and\nthe time point of the wave.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Note, that it would be very difficult to derive the likelihood function of the stochastic SIR model for this kind of observations. Therefore this inference problem is a good fit for ABC algorithms.","category":"page"},{"location":"example/#Prior","page":"Example","title":"Prior","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"For Bayesian inference we also need to define a prior distribution for the parameters (assuming independence):","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Transmission rate: beta sim textUniform(01 1)\nRecovered rate: gamma sim textUniform(005 05)","category":"page"},{"location":"example/#Implementation","page":"Example","title":"Implementation","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"First we load the required packages and set a random seed.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"using SimulatedAnnealingABC\nusing Distributions\nusing Plots\n\nimport Random\nRandom.seed!(123)\nnothing # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Then we define the stochastic SIR model","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"function stochastic_SIR(θ; S0::Int, I0::Int, R0::Int, t_max)\n\n    β, γ = θ\n    # Initialize variables\n    S = S0\n    I = I0\n    R = R0\n    t = 0.0\n    N = S0 + I0 + R0  # Total population\n\n    # Records of the simulation\n    time = [t]\n    S_record = [S]\n    I_record = [I]\n    R_record = [R]\n\n    while t < t_max && I > 0\n        # Calculate rates\n        infection_rate = β * S * I / N\n        recovery_rate = γ * I\n        total_rate = infection_rate + recovery_rate\n\n        # Time to next event\n        dt = rand(Exponential(1 / total_rate))\n        t += dt\n\n        # Determine which event occurs\n        if rand() < infection_rate / total_rate\n            # Infection event\n            S -= 1\n            I += 1\n        else\n            # Recovery event\n            I -= 1\n            R += 1\n        end\n\n        # Record the state\n        push!(time, t)\n        push!(S_record, S)\n        push!(I_record, I)\n        push!(R_record, R)\n    end\n\n\n    return (time = time, S = S_record,\n            I = I_record, R = R_record)\nend\nnothing  # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"For the sake of this example we simulate data. The random seed has a large influence on how the simulation result looks like.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"# \"true\" parameters\nθtrue = [0.3,    # Transmission rate β\n         0.1]    # Recovery rate γ\n\nsim = stochastic_SIR(θtrue; S0=99, I0=1, R0=0, t_max=160)\n\n# Plot the results\nplot(sim.time, sim.S, label=\"Susceptible\", xlabel=\"Time (days)\", ylabel=\"Number of Individuals\",\n     title=\"Stochastic SIR Model\", linewidth=2, linetype=:steppre)\nplot!(sim.time, sim.I, label=\"Infected\", linewidth=2, linetype=:steppre)\nplot!(sim.time, sim.R, label=\"Recovered\", linewidth=2, linetype=:steppre)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The observation are defined as follows:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"data_obs = (\n    total_infected = sim.R[end],\n    peak_infected = maximum(sim.I),\n    t_peak = sim.time[argmax(sim.I)]\n)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"To use sabc we need to define a function that returns the distance(s) between a random simulation of our with parameter theta and the observed data. Here we define two functions, the first one returns the distance of each observed statistics and the second one aggregates this into a single distance.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"function f_dist_multi_stats(θ, data_obs)\n    # run model\n    sim = stochastic_SIR(θ; S0=99, I0=1, R0=0, t_max=160)\n\n    sim_total_infected = sim.R[end]\n    sim_peak_infected = maximum(sim.I)\n    sim_t_peak = sim.time[argmax(sim.I)]\n\n    # compute distance of summary statistics\n\n    dists = (abs2(sim_total_infected - data_obs.total_infected),\n             abs2(sim_peak_infected - data_obs.peak_infected),\n             abs2(sim_t_peak - data_obs.t_peak))\n\nend\n\n# For single stat version we need to aggregate the statistics.\n# Here we give the same weight to each statistic.\nf_dist_single_stat(θ, data_obs) = sum(f_dist_multi_stats(θ, data_obs))\nnothing # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The prior is defined with Distributions.jl:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"prior = product_distribution([Uniform(0.1, 1),     # β\n                              Uniform(0.05, 0.5)]) # γ","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"With everything in place, we run the inference with the :single_eps and :multi_eps algorithm. Note, that the argument data_obs is passed to the distance functions.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"res_1 = sabc(f_dist_single_stat, prior, data_obs;\n             n_simulation = 200_000,\n             n_particles = 5000,\n             algorithm = :single_eps)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"res_2 = sabc(f_dist_multi_stats, prior, data_obs;\n             n_simulation = 200_000,\n             n_particles = 5000,\n             algorithm = :multi_eps)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Finally, we plot the posterior parameter distribution. If a larger number of iterations is used, the difference between the two algorithm would become smaller.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"pop_1 = stack(res_1.population)\npop_2 = stack(res_2.population)\n\np1 = scatter(pop_1[1,:], pop_1[2,:], title=\"single stat\",\n             markeralpha = 0.2,\n             markersize = 1,\n             markerstrokewidth = 0);\nscatter!(p1, θtrue[1:1], θtrue[2:2], markersize = 2);\n\np2 = scatter(pop_2[1,:], pop_2[2,:], title=\"multible stats\",\n             markeralpha = 0.2,\n             markersize = 1,\n             markerstrokewidth = 0);\nscatter!(p2, θtrue[1:1], θtrue[2:2], markersize = 2);\n\nplot(p1, p2, xlab = \"β\", ylab = \"γ\", legend=false)","category":"page"}]
}
