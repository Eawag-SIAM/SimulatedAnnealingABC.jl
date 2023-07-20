## -------------------------------------------------------
## Functions to learn (multivariate) cdf
## -------------------------------------------------------

import StatsBase
using Interpolations: interpolate, extrapolate,
    LinearMonotonicInterpolation, SteffenMonotonicInterpolation, Flat

# -----------
# 1-dim

# """
# Estimate the cdf of data `x`.
# Returns a function.
# """
# function build_cdf(x::AbstractVector)
#     StatsBase.ecdf(x)
# end


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
# 1-dim



"""
Construct the empirical cdf from a multidimensional data

It assumes every columns `x` coresponds to an sample.

Returns a function
"""
function build_cdf(x::Array{T, 2} where T)
    function(z)
        n, d = size(x)
        length(z) == d || error("Expect an input vector of length $(d)!")
        count = 0
        for i in 1:n
            k = 1
            # check each if in all dimensions z <= x
            for j in 1:d
                if x[i,j] > z[j]
                    k = 0
                    break
                end
            end
            count += k
        end
        count / n
    end
end
