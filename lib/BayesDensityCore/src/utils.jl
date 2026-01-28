# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Logit map
logit(x::Real) = log(x / (1-x))

xlogx(x::Real) = ifelse(x > 0.0, x*log(x), 0.0)

# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# Count the number of times each integer from 1:K occurs in the array `z`
function countint(z::AbstractVector{<:Integer}, K::Int)
    counts = zeros(Int, K)
    for i in eachindex(z)
        counts[z[i]] += 1
    end
    return counts
end

# Map an unconstrained K-1 dimensional vector to the K-simplex through logistic stickbreaking, defined as the composition of the logistic and stickbreaking maps.
# To ensure numerical stability, the calculation is performed in log-space.
function logistic_stickbreaking(β::AbstractVector{T}) where {T<:Real}
    K = length(β) + 1
    log_π = Vector{T}(undef, K)
    softplus_sum = zero(T)
    for k in 1:K-1
        softplus_sum += softplus(β[k])
        log_π[k] = β[k] - softplus_sum
    end
    log_π[K] = -softplus_sum
    return softmax(log_π)
end

# Stickbreaking map. Takes a vector in [0, 1]^(K-1) and outputs a simplex-vector
# NB! It is implicit that V[K] = 1 in the input here.
function truncated_stickbreaking(v::AbstractVector{T}) where {T<:Real}
    K = length(v)+1
    w = Vector{T}(undef, K)
    w[1] = v[1]
    cum_w = w[1]
    for k in 2:K-1
        w[k] = v[k] * (1 - cum_w)
        cum_w += w[k]
    end
    w[K] = 1 - cum_w
    return w
end

# Compute bin counts on a regular grid consisting of `M` bins over the interval [xmin, xmax]
function bin_regular(x::AbstractVector{T}, xmin::T, xmax::T, M::Int, right::Bool) where {T<:Real}
    R = xmax - xmin
    bincounts = zeros(Int, M)
    edges_inc = M/R
    if right
        for val in x
            idval = min(M-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            bincounts[idval] += 1.0
        end
    end
    return bincounts
end

function bin_irregular(x::AbstractVector{<:Real}, breaks::AbstractVector{<:Real}, right::Bool)
    K = length(breaks)-1
    bincounts = zeros(Int64, length(breaks)-1)
    if right
        for val in x
            idval = min(max(1, searchsortedfirst(breaks, val) - 1), K)
            bincounts[idval] += 1
        end
    else
        for val in x
            idval = max(min(K, searchsortedlast(breaks, val)), 1)
            bincounts[idval] += 1
        end
    end
    return bincounts
end

# Compute linear binning on a regular grid
function linear_binning(x::AbstractVector{T}, n_bins::Int, xmin::T, xmax::T) where {T<:Real}
    N = zeros(T, n_bins)
    delta = (xmax - xmin) / n_bins

    for i in eachindex(x)
        lxi = ((x[i] - xmin) / delta) + 1
        li = floor(Int, lxi)
        rem = lxi - li

        if 1 <= li < n_bins
            N[li] += 1 - rem
            N[li + 1] += rem
        end
    end
    return round.(Int, N)
end

# Create the k'th unit vector in the canonical basis for R^K.
function unitvector(K::Int, k::Int, T::Type{<:Real}=Float64)
    if !(1 ≤ k ≤ K)
        throw(ArgumentError("Index out of range."))
    end
    unitvec = zeros(T, K)
    unitvec[k] = 1
    return unitvec
end