# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Logit map
logit(x::Real) = log(x / (1-x))

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

# Map an unconstrained K-1 dimensional vector to the K-simplex through stickbreaking
# To ensure numerical stability, the calculation is performed in log-space.
function stickbreaking(β::AbstractVector{T}) where {T<:Real}
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

# Create the k'th unit vector in the canonical basis for R^K.
function unitvector(K::Int, k::Int, T::Type{<:Real}=Float64)
    if !(1 ≤ k ≤ K)
        throw(ArgumentError("Index out of range."))
    end
    unitvec = zeros(T, K)
    unitvec[k] = 1
    return unitvec
end