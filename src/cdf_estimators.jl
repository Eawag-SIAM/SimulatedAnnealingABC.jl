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
    n = length(x)
    # note, we add a 0 and 1 probability points
    probs = [0; [(k - 0.5)/n for k in 1:n]; 1]
    a = 1.5
    values = [0; sort(x); maximum(x)*a]

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
    cdfs = [build_cdf(xi) for xi in eachcol(x)]

    # construct function
    function f(ρ)
        [cdfs[i](ρ[i]) for i in 1:length(ρ)]
    end

    return f
end
