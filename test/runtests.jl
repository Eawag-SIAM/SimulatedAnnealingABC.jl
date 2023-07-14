using SimulatedAnnealingABC
using Test
using Distributions


@testset "Sampling 1-dim" begin

    @testset "1 summary statistic" begin
        ## Define model
        f_dist(θ) = abs(0.0 - rand(Normal(θ[1], 1)))
        prior = Uniform(-10,10)

        # n_simulation too small
        @test_throws ErrorException sabc(f_dist, prior; eps_init = 1,
                                         n_particles = 100, n_simulation = 10)


        res = sabc(f_dist, prior; eps_init = 10,
                   n_particles = 100, n_simulation = 1000)

        @test res.state.n_simulation <= 1000
        @test length(res.population) == 100

        ## update existing population
        update_population!(res, f_dist, prior;
                           n_simulation = 1_000)

        @test res.state.n_simulation <= 1000 + 1000

        ## update existing population with too few simulation, i.e.
        ##  n_simulation < n_particles
        n_sim = res.state.n_simulation
        update_population!(res, f_dist, prior;
                           n_simulation = 50)
        @test res.state.n_simulation == n_sim # no updating

    end

    @testset "n summary statistics" begin

        ## returns vector of summary stats
        function f_dist_vector(θ)
            y_sim = rand(Normal(θ[1], 1))
            [abs(0.0 - y_sim), abs2(0.0 - y_sim)]
        end
        ## returns tuple of summary stats
        function f_dist_tuple(θ)
            y_sim = rand(Normal(θ[1], 1))
            (abs(0.0 - y_sim), abs2(0.0 - y_sim))
        end

        prior = Uniform(-10,10)

        # single eps_init
        for eps_init in [10, 10.0, (10, 10), [10, 10.0]]
            @test length(sabc(f_dist_vector, prior; eps_init = eps_init,
                              n_particles = 100, n_simulation = 1000).state.ϵ) == 2
            @test length(sabc(f_dist_tuple, prior; eps_init = eps_init,
                              n_particles = 100, n_simulation = 1000).state.ϵ) == 2
        end
    end

end


@testset "Sampling n-dim" begin

    ## Define model
    f_dist(θ) = sum(abs2, rand(Normal(θ[1], θ[2]), 4))
    prior = product_distribution([Normal(0,1),   # theta[1]
                                  Uniform(0,1)]) # theta[2]

    # n_simulation too small
    @test_throws ErrorException sabc(f_dist, prior; eps_init = 1,
                                     n_particles = 100, n_simulation = 10)


    res = sabc(f_dist, prior; eps_init = 10,
               n_particles = 100, n_simulation = 1000)

    @test res.state.n_simulation <= 1000
    @test length(res.population) == 100

    ## update existing population
    update_population!(res, f_dist, prior;
                       n_simulation = 1_000)

    @test res.state.n_simulation <= 1000 + 1000

    ## update existing population with too few simulation, i.e.
    ##  n_simulation < n_particles
    n_sim = res.state.n_simulation
    update_population!(res, f_dist, prior;
                       n_simulation = 50)
    @test res.state.n_simulation == n_sim # no updating

end

@testset "Convergence" begin
    # TOOO
end


@testset "Distance transformation" begin
    # TOOO
end
