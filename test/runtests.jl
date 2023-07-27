using SimulatedAnnealingABC
using Test
using Distributions


@testset "Sampling 1-dim" begin

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

@testset "Parallization" begin
    using Base.Threads: nthreads, @threads, @spawn
    using Base.Iterators: partition
    using Ranges

    tasks_per_thread = 2
    chunk_size = max(1, length(some_data) ÷ (tasks_per_thread * nthreads()))
    data_chunks = partition(some_data, chunk_size)
    f(i) = (sleep(0.001); i);
    tasks = map(data_chunks) do chunk
        # Each chunk of your data gets its own spawned task that does its own local, sequential work
        # and then returns the result
        @spawn begin
            state = 0
            for x in chunk
                state += f(x)
            end
            return state
        end
    end
    states = fetch.(tasks) # get all the values returned by the individual tasks. fetch is type
                           # unstable, so you may optionally want to assert a specific return type.
    
    @test states[1] == 1275 
    print(states)
end