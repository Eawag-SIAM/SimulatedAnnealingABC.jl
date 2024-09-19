using SimulatedAnnealingABC
using Test
using Distributions
using Logging

ENV["CI"] = true                # to disable progress bar
global_logger(ConsoleLogger(stderr, Logging.Warn)) # disable logging

@testset "cdf estimator" begin

    x = rand(100) .* 4
    cdf = SimulatedAnnealingABC.build_cdf(x)
    @test cdf(0) <= sqrt(eps())
    @test cdf(Inf) ≈ 1
    @test all(diff(cdf(sort(rand(100)))) .>= 0)  # is cdf is monotonous?

    # includes repetitions
    cdf = SimulatedAnnealingABC.build_cdf([1, 2, 2, 3, 3, 3])
    @test cdf(0) <= sqrt(eps())
    @test cdf(Inf) ≈ 1
    @test all(diff(cdf(sort(rand(100).*3))) .>= 0)

    # inludes zero
    cdf = SimulatedAnnealingABC.build_cdf([1, 0, 2, 0, 3])
    @test cdf(0) <= sqrt(eps())
    @test cdf(Inf) ≈ 1
    @test all(diff(cdf(sort(rand(100).*3))) .>= 0)

end

@testset "Single summary statistics" verbose = true begin
    @testset "Sampling 1-dim" begin

        ## Define model
        f_dist(θ) = abs(0.0 - mean(rand(Normal(θ[1], 1), 100)))
        prior = Uniform(-10,10)

        # n_simulation too small
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10)

        # illegal tuning parameter
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10,
                                         v = -0.1)
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10,
                                         β = -0.1)
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10,
                                         β = 1.1)
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10,
                                         δ = -0.1)

        for algorithm in [:multi_eps, :single_eps]

            res = sabc(f_dist, prior;
                       n_particles = 100, n_simulation = 1000,
                       algorithm = algorithm)

            @test res.state.n_simulation <= 1000
            @test res.state.n_population_updates == 9
            @test length(res.population) == 100

            ## update existing population
            update_population!(res, f_dist, prior;
                               n_simulation = 1_000)

            @test res.state.n_simulation <= 2000
            @test res.state.n_population_updates == 19

            ## update existing population with too few simulation, i.e.
            ##  n_simulation < n_particles
            n_sim = res.state.n_simulation
            update_population!(res, f_dist, prior;
                               n_simulation = 50)
            @test res.state.n_simulation == n_sim # no updating
        end
    end


    @testset "Sampling n-dim" begin

        ## Define model
        f_dist(θ) = abs(0.0 - mean(rand(Normal(θ[1], θ[2]), 100)))
        prior = product_distribution([Normal(0,1),   # theta[1]
                                      Uniform(0,1)]) # theta[2]

        # n_simulation too small
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100,
                                         n_simulation = 10)

        for algorithm in [:multi_eps, :single_eps]

            res = sabc(f_dist, prior;
                       n_particles = 100, n_simulation = 1000,
                       algorithm = algorithm)

            @test res.state.n_simulation <= 1000
            @test length(res.population) == 100

            ## update existing population
            update_population!(res, f_dist, prior;
                               n_simulation = 1_000)

            @test res.state.n_simulation <= 2000

            ## update existing population with too few simulation, i.e.
            ##  n_simulation < n_particles
            n_sim = res.state.n_simulation
            update_population!(res, f_dist, prior;
                               n_simulation = 50)
            @test res.state.n_simulation == n_sim # no updating
        end

    end

end

@testset "Multible summary statistics" verbose = true begin
    @testset "Sampling 1-dim" begin
        # define prior
        prior = Normal(0,1)

        # compte two summary statistics
        function f_dist(θ)
            y_samp = rand(Normal(θ[1], 1), 10)
            (abs(0 - mean(y_samp)), abs(1 - mean(y_samp .^2)))
        end

        for algorithm in [:multi_eps, :single_eps]

            ## Sample Posterior
            res = sabc(f_dist, prior;
                       n_particles = 100, n_simulation = 1000,
                       algorithm = algorithm);

            @test all(res.state.ϵ .< 1)
            @test res.state.n_simulation <= 1000
            @test length(res.population) == 100

            ## update existing population
            update_population!(res, f_dist, prior;
                               n_simulation = 1_000)

            @test res.state.n_simulation <= 2000

            ## update existing population with too few simulation, i.e.
            ##  n_simulation < n_particles
            n_sim = res.state.n_simulation
            update_population!(res, f_dist, prior;
                               n_simulation = 50)
            @test res.state.n_simulation == n_sim # no update
        end

    end

    @testset "Sampling n-dim" begin

        # define prior
        prior = product_distribution([Normal(0,1),   # theta[1]
                                      Uniform(0,2)]) # theta[2]

        # compte two summary statistics
        function f_dist(θ)
            y_samp = rand(Normal(θ[1], θ[2]), 10)
            (abs(0 - mean(y_samp)), abs(1 - mean(y_samp .^2)))
        end

        for algorithm in [:multi_eps, :single_eps]

            ## Sample Posterior
            res = sabc(f_dist, prior;
                       n_particles = 100, n_simulation = 1000,
                       algorithm = algorithm);

            @test all(res.state.ϵ .< 1)

            @test res.state.n_simulation <= 1000
            @test length(res.population) == 100

            ## update existing population
            update_population!(res, f_dist, prior;
                               n_simulation = 1_000)

            @test res.state.n_simulation <= 2000

            ## update existing population with too few simulation, i.e.
            ##  n_simulation < n_particles
            n_sim = res.state.n_simulation
            update_population!(res, f_dist, prior;
                               n_simulation = 50)
            @test res.state.n_simulation == n_sim # no update
        end
    end
end
