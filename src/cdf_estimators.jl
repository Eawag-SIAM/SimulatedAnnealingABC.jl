## -------------------------------------------------------
## Functions to learn (multivariate) cdf
## -------------------------------------------------------

import StatsBase
using Interpolations: interpolate, extrapolate,
    LinearMonotonicInterpolation, SteffenMonotonicInterpolation, Flat

# -----------
# 1-dim


"""
Estimate the empirical cdf of data `x`
smoothed by interpolation.

Returns a function.
"""
function build_cdf(x::AbstractVector)
    # here x is a coulmn of the prior distance matrix
    # i.e., all distances (for all particles) for one given stat
    n = length(x)       # n = the number of particles
    # this is the y-axis for the interpolation:
    # note, we add a 0 and 1 probability points
    probs = [0; [(k - 0.5)/n for k in 1:n]; 1] 
    # this is the x-axis, including 0 and 1.5*(largest distance): 
    a = 1.5
    values = [0; sort(x); maximum(x)*a]

    # returns a function that given a distance, returns a value between 0 and 1 
    extrapolate(interpolate(values, probs,
                            LinearMonotonicInterpolation()
                            #SteffenMonotonicInterpolation()
                            ),
                Flat())

end


# -----------
# n-dim


"""
Construct the empirical cdf for each summary statistics

It assumes every row `x` corresponds to a sample.

Returns a function, that applies the corresponding cdf to each statistics.
"""
function build_cdf(x::Array{T, 2} where T)
    # build 1d cdfs
    # NOTE: x is the 'distances_prior' (n_particles x n_stats) matrix
    #       eachcol(x) selects all distances (for all particles) for one given stat
    #       'build_cdf' constructs cdf functions for each summary stat
    cdfs = [build_cdf(xi) for xi in eachcol(x)]

    # construct and return function
    # ρ is a vector, a row of the distance matrix, with size = number of stats
    # (distances for all stats, for a given particle)  
    function f(ρ)
        [cdfs[i](ρ[i]) for i in 1:length(ρ)]
    end

    return f
end
