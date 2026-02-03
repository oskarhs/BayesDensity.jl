function linear_binning(x::AbstractVector{T}, n_bins::Int, xmin::T, xmax::T) where {T<:Real}
    N = zeros(T, n_bins)
    delta = (xmax - xmin) / n_bins
    
    # Compute first midpoint
    mid_first = delta / 2

    for i in eachindex(x)
        # Intervals 0 and n_bins are valid
        lxi = ((x[i] - mid_first) / delta) + 1
        li = floor(Int, lxi)
        rem = lxi - li

        if 1 <= li < n_bins
            N[li] += 1 - rem
            N[li + 1] += rem
        elseif li == n_bins
            N[end] += one(T)
        else
            N[begin] += one(T)
        end
    end
    return round.(Int, N)
end

# Compute the normalization constant of `shs` for given parameters using Simpson's method
# NB! Computes the normalization constant on [0, 1] and not on the original scale.
function compute_norm_constants_cdf_grid(shs::HistSmoother{T}, params::NamedTuple{Names, Vals}) where {T<:Real, Names, Vals<:Tuple}
    kn = knots(shs.bs)
    n = 4000
    # Evaluation grid for Simpson's method
    eval_grid = LinRange{T}(kn[begin], kn[end], n+1)
    h = step(eval_grid)
    val_cdf = Vector{T}(undef, div(n, 2)+1)

    val_cdf[1] = zero(T)
    y = exp.(eval_linpred(shs, params, eval_grid))
    for i in 2:length(val_cdf)
        val_cdf[i] = h/3 * (y[2*(i-2)+1] + 4*y[2*(i-1)] + y[2*(i-1)+1]) + val_cdf[i-1]
    end
    l1_norm = val_cdf[end]
    val_cdf = val_cdf / l1_norm
    return eval_grid[1:2:end], val_cdf, l1_norm
end
function compute_norm_constants_cdf_grid(shs::HistSmoother{T}, params::AbstractVector{NamedTuple{Names, Vals}}) where {T<:Real, Names, Vals<:Tuple}
    kn = knots(shs.bs)
    bounds = shs.data.bounds
    n = 4000
    # Evaluation grid for Simpson's method
    eval_grid = LinRange{T}(kn[begin], kn[end], n+1)
    h = step(eval_grid)
    val_cdf = Vector{Vector{T}}(undef, length(params))

    y = exp.(eval_linpred(shs, params, eval_grid)) # (length(t), length(samples)) matrix
    l1_norms = Vector{T}(undef, length(params))
    for j in eachindex(params)
        val_cdf_j = Vector{T}(undef, div(n, 2)+1)
        val_cdf_j[1] = zero(T)
        for i in 2:length(val_cdf_j)
            val_cdf_j[i] = h/3 * (y[2*(i-2)+1, j] + 4*y[2*(i-1), j] + y[2*(i-1)+1, j]) + val_cdf_j[i-1]
        end
        l1_norms[j] = val_cdf_j[end]
        val_cdf[j] = val_cdf_j / l1_norms[j]
    end
    return eval_grid[1:2:end], val_cdf, l1_norms
end

function eval_linpred(shs::HistSmoother{T}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}) where {T<:Real, Names, Vals, S<:Real}
    Z = demmler_reinsch_basis_matrix(t, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t)), t, Z)

    # Compute linear predictor
    linpreds = C * params.β
    return linpreds
end
function eval_linpred(shs::HistSmoother{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}) where {T<:Real, Names, Vals, S<:Real}
    R = promote_type(T, S)
    # Reshape beta parameters into a Matrix, l1 norms into a vector
    β_mat = Matrix{R}(undef, (length(shs.bs), length(params)))
    for i in eachindex(params)
        β_mat[:,i] = params[i].β
    end

    Z = demmler_reinsch_basis_matrix(t, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t)), t, Z)

    # Compute linear predictor
    linpreds = C * β_mat
    return linpreds
end

function demmler_reinsch_basis_matrix(x::AbstractVector{<:Real}, bs::AbstractBSplineBasis{N, <:Real}, LZ::AbstractMatrix{<:Real}) where {N}
    # Get B-spline basis matrix
    B = create_unnormalized_sparse_spline_basis_matrix(x, bs)

    # Mixed model reparametrization
    Z = B * LZ
    return Z
end

function create_unnormalized_sparse_spline_basis_matrix(x::AbstractVector{T}, basis::AbstractBSplineBasis{N, S}) where {T<:Real, N, S<:Real}
    K = length(basis)

    n = length(x)

    I = Vector{Int}(undef, 4*n) # row indices
    J = Vector{Int}(undef, 4*n) # column indices
    V = Vector{promote_type(T, S)}(undef, 4*n)
    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in eachindex(x)
        ind = (4*i-3):(4*i)
        j, basis_eval = basis(x[i])
        I[ind] .= i
        j = max(4, j)
        J[ind] .= (j-3):j
        V[ind] .= reverse(basis_eval) 
    end
    return sparse(I, J, V, n, K)
end