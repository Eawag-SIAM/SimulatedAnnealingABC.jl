## -------------------------------------------------------
## Functions to learn (multivariate) cdf
## -------------------------------------------------------

import StatsBase
using Interpolations: interpolate, extrapolate,
    LinearMonotonicInterpolation, Flat

# -----------
# 1-dim


"""
Estimate the empirical cdf of data `x`
smoothed by interpolation.

**Note**, do not use this function outside of this package! It does not
return a correct cdf estimation if x contains duplicates. (Which
should be fine for the purpose here.)

Returns a function.
"""
function build_cdf(x::AbstractVector)
    # here x is a column of the prior distance matrix
    # i.e., all distances (for all particles) for one given stat

    # this is the x-axis, including 0 and 1.5*(largest distance):
    # we drop zeros because the `interpolate` cannot handle multiple observation at zero.
    x = filter(e -> e > 0, x)

    # add a single zero observation and a larger max observation
    a = 1.5
    values = [0; sort(x); maximum(x)*a]

    # this is the y-axis for the interpolation:
    probs = range(0, stop=1, length=length(values))

    # returns a function that given a distance computes a value between 0 and 1
    extrapolate(interpolate(values, probs,
                            LinearMonotonicInterpolation()
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
    #       'build_cdf' constructs a cdf functions for each summary stat
    cdfs = [build_cdf(xi) for xi in eachcol(x)]

    # construct and return function
    # ρ is a vector, a row of the distance matrix, with size = number of stats
    # (distances for all stats, for a given particle)
    function f(ρ)
        [cdfs[i](ρ[i]) for i in eachindex(ρ)]
    end

    return f
end
