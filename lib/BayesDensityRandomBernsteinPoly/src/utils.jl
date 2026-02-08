# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# Compute bin counts on a regular grid consisting of `M` bins over the interval [xmin, xmax]
function bin_regular(x::AbstractVector{T}, xmin::T, xmax::T, n_bins::Int) where {T<:Real}
    R = xmax - xmin
    bincounts = zeros(Int, n_bins)
    edges_inc = n_bins/R
    for val in x
        idval = min(n_bins-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
        bincounts[idval] += 1
    end
    return bincounts
end

function bin_regular_ind(y::AbstractVector{T}, xmin::T, xmax::T, n_bins::Int) where {T<:Real}
    R = xmax - xmin
    bin_inds = zeros(Int, length(y))
    edges_inc = n_bins/R
    for i in eachindex(y)
        idval = min(n_bins-1, floor(Int, (y[i]-xmin)*edges_inc+eps())) + 1
        bin_inds[i] = idval
    end
    return bin_inds
end