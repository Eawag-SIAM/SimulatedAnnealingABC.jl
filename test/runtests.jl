using SimulatedAnnealingABC
using Test
using Distributions


@testset "Single summary statistics" begin
    @testset "Sampling 1-dim" begin

        ## Define model
        f_dist(θ) = abs(0.0 - rand(Normal(θ[1], 1)))
        prior = Uniform(-10,10)

        # n_simulation too small
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100, n_simulation = 10)


        res = sabc(f_dist, prior;
                   n_particles = 100, n_simulation = 1000)

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


    @testset "Sampling n-dim" begin

        ## Define model
        f_dist(θ) = sum(abs2, rand(Normal(θ[1], θ[2]), 4))
        prior = product_distribution([Normal(0,1),   # theta[1]
                                      Uniform(0,1)]) # theta[2]

        # n_simulation too small
        @test_throws ErrorException sabc(f_dist, prior;
                                         n_particles = 100,
                                         n_simulation = 10)


        res = sabc(f_dist, prior;
                   n_particles = 100, n_simulation = 1000)

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

@testset "Multible summary statistics" begin
    @testset "Sampling 1-dim" begin
        # TOOO
    end

    @testset "Sampling n-dim" begin

        # define prior
        prior = product_distribution([Normal(0,1),   # theta[1]
                                      Uniform(0,2)]) # theta[2]

        # compte two summary statistics
        function f_dist(θ, y)
            y_samp = rand(Normal(θ[1], θ[2]), 10)
            (abs(0 - sum(y_samp .- 0)), abs(sum(1 .- (y_samp .- 0).^2)))
        end


        y_obs = randn(10)

        ## Sample Posterior
        res = sabc(f_dist, prior, y_obs;
                   n_particles = 1000, n_simulation = 10_000);

        @test res.state.ϵ < 1

    end
end

@testset "Convergence" begin
    # TOOO
end


@testset "Distance transformation" begin
    # TOOO
end
