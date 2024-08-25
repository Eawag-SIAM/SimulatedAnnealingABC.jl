var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"sabc","category":"page"},{"location":"api/#SimulatedAnnealingABC.sabc","page":"API","title":"SimulatedAnnealingABC.sabc","text":"sabc(f_dist::Function, prior::Distribution, args...;\n      n_particles = 100, n_simulation = 10_000,\n      type = :multi,\n      resample = 2*n_particles,\n      v=1.0, β=0.8, δ=0.1,\n      checkpoint_history = 1,\n      show_progressbar::Bool = !is_logging(stderr),\n      show_checkpoint = is_logging(stderr) ? 100 : Inf,\n      kwargs...)\n\nSimulated Annealing Approximate Bayesian Inference Algorithm\n\nArguments\n\nf_dist: Function that one or more distances between data and a random sample from the likelihood. The first argument must be the parameter vector.\nprior: A Distribution defining the prior.\nargs...: Further arguments passed to f_dist\nn_particles: Desired number of particles.\nn_simulation: maximal number of simulations from f_dist.\nv = 1.0: Tuning parameter for XXX\nβ = 0.8: Tuning parameter for XXX\nδ = 0.1: Tuning parameter for XXX\ntype = :multi: Choose algorithm, either :multi, or :hybrid\nresample: After how many accepted population updates?\ncheckpoint_history = 1: every how many population updates distances and epsilons are stored\nshow_progressbar::Bool = !is_logging(stderr): defaults to true for interactive use.\nshow_checkpoint::Int = 100: every how many population updates algorithm state is displayed.                               By default disabled for for interactive use.\nkwargs...: Further arguments passed to f_dist`\n\nReturn\n\nAn object of type SABCresult\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"update_population!","category":"page"},{"location":"api/#SimulatedAnnealingABC.update_population!","page":"API","title":"SimulatedAnnealingABC.update_population!","text":"update_population!(population_state::SABCresult,\n                   f_dist, prior, args...;\n                   n_simulation,\n                   v=1.0, β=0.8, δ=0.1,\n                   resample = 2*length(population_state.population),\n                   checkpoint_history = 1,\n                   show_progressbar::Bool = !is_logging(stderr),\n                   show_checkpoint = is_logging(stderr) ? 100 : Inf,\n                   kwargs...)\n\nUpdates particles with n_simulation and applies importance sampling if needed. Modifies population_state.\n\nArguments\n\nSee docstring for sabc.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"SimulatedAnnealingABC.SABCresult","category":"page"},{"location":"api/#SimulatedAnnealingABC.SABCresult","page":"API","title":"SimulatedAnnealingABC.SABCresult","text":"Holds results from a SABC run with fields:\n\npopulation: vector of parameter samples from the approximate posterior\nu: transformed distances\nρ: distances\nstate: state of algorithm\n\nThe history of ϵ can be accessed with the field state.ϵ_history.\n\n\n\n\n\n","category":"type"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/#Getting-started","page":"Usage","title":"Getting started","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"A minimal example how to use this package for inference:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using SimulatedAnnealingABC\nusing Distributions\n\n# Define a stochastic model.\n# Your real model should be so complex, that it would be too\n# complicated to compute it's likelihood function.\nfunction my_stochastic_model(θ, n)\n    rand(Normal(θ[1], θ[2]), n)\nend\n\n# define prior of the parameters\nprior = product_distribution([Normal(0,2),   # theta[1]\n                              Uniform(0,2)]) # theta[2]\n\n\n# Simulate some observation data\ny_obs = rand(Normal(2, 0.5), 10)\n\n# Define a function that simulate with my_stochastic_model and then\n# measure the distances of the simulated and the observed data with\n# two summary statistics\nfunction f_dist(θ; y_obs)\n    y_sim = my_stochastic_model(θ, length(y_obs))\n\n    (\n        abs(mean(y_obs) - mean(y_sim)),\n        abs(sum(abs2, y_obs) - sum(abs2, y_sim))\n    )\nend\n\n\n## Sample Posterior\nres = sabc(f_dist, prior;\n           n_particles = 1000, n_simulation = 100_000, y_obs=y_obs)\n\n\n## Improve the results by running the inference for longer\nres2 = update_population!(res, f_dist, prior;\n                          n_simulation = 50_000, y_obs=y_obs)\n","category":"page"},{"location":"usage/#Logging-and-progress-bar","page":"Usage","title":"Logging and progress bar","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"There are two ways sabc can inform about the progress of the ongoing inference run:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Progress bar. By default it is shown in interactive sessions. It can be disabled with the argument (show_progressbar = false)\nLogging statements. The argument show_checkpoint controls how often a summary of the current state is logged. For long running computations in a cluster environment this is more convenient than the progress bar.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"You can set the logging level to Warn to suppress all logging statements:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using Logging\nglobal_logger(ConsoleLogger(stderr, Logging.Warn))\n\n... run sabc() ...","category":"page"},{"location":"usage/#How-to-disable-multi-threading","page":"Usage","title":"How to disable multi-threading","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"TODO","category":"page"},{"location":"related/#Related-Packages","page":"Related packages","title":"Related Packages","text":"","category":"section"},{"location":"related/#Julia","page":"Related packages","title":"Julia","text":"","category":"section"},{"location":"related/#Approximate-Bayesian-Computation","page":"Related packages","title":"Approximate Bayesian Computation","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"ABCdeZ.jl - Approximate Bayesian Computation (ABC) with differential evolution (de) moves and model evidence (Z) estimates.\nApproxBayes.jl - Implements basic ABC rejection sampler and sequential monte carlo algorithm (ABC SMC) as in Toni. et al 2009 as well as model selection versions of both (Toni. et al 2010).\nGpABC.jl - ABC with emulation on Gaussian Process Regression.\nKissABC.jl - Implementation of Multiple Affine Invariant Sampling for efficient Approximate Bayesian Computation. Is looking for a new maintainer.","category":"page"},{"location":"related/#Misc","page":"Related packages","title":"Misc","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"SimulationBasedInference.jl Despite the name this package seems to focus on traditional Bayesian inference methods and not ABC, i.e. it assumes we can evaluate the density of the posterior.","category":"page"},{"location":"related/#R","page":"Related packages","title":"R","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"abc","category":"page"},{"location":"related/#Python","page":"Related packages","title":"Python","text":"","category":"section"},{"location":"related/","page":"Related packages","title":"Related packages","text":"pyABC","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"CurrentModule = SimulatedAnnealingABC","category":"page"},{"location":"#SimulatedAnnealingABC","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"","category":"section"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"Documentation for SimulatedAnnealingABC.","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"This package provides different SimulatedAnnealingABC (SABC) algorithms for Approximate Bayesian Computation (ABC) (sometimes also called simulation-based inference or likelihood-free inference).","category":"page"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"note: Note\nCan you evaluate the density of your posterior? Then you should most likely not be using this or any other ABC package! Conventional MCMC algorithm will be much more efficient.","category":"page"},{"location":"#References","page":"SimulatedAnnealingABC","title":"References","text":"","category":"section"},{"location":"","page":"SimulatedAnnealingABC","title":"SimulatedAnnealingABC","text":"Albert, C., Künsch, H.R., Scheidegger, A., 2015. A simulated annealing approach to approximate Bayes computations. Statistics and computing 25, 1217–1232. https://doi.org/10.1007/s11222-014-9507-8","category":"page"},{"location":"example/#Worked-example","page":"Worked Example","title":"Worked example","text":"","category":"section"},{"location":"example/","page":"Worked Example","title":"Worked Example","text":"TODO","category":"page"},{"location":"example/","page":"Worked Example","title":"Worked Example","text":"Translate and update example from readme.","category":"page"},{"location":"example/","page":"Worked Example","title":"Worked Example","text":"demonstrate pro and cons of the different algo types.","category":"page"}]
}