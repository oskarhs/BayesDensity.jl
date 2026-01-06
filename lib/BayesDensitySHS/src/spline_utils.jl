# Compute the normalization constant of `shs` for given parameters using Simpson's method
# NB! Computes the normalization constant on [0, 1] and not on the original scale.
function compute_norm_constants(shs::SHSModel{T, A, D}, params::NamedTuple{Names, Vals}) where {T<:Real, A, D, Names, Vals<:Tuple}
    kn = knots(shs.bs)
    n = 2048
    # Evaluation grid for Simpson's method
    t = LinRange{T}(kn[1], kn[end], n+1)
    h = step(t)

    y = exp.(eval_linpred(shs, params, t)) # (length(t), length(samples)) matrix
    l1_norm = h / 3 * (y[1] + 4*sum(y[2:2:n]) + 2 * sum(y[3:2:n-1]) + y[n+1])
    return l1_norm
end
function compute_norm_constants(shs::SHSModel{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}) where {T<:Real, A, D, Names, Vals<:Tuple}
    kn = knots(shs.bs)
    n = 2048
    # Evaluation grid for Simpson's method
    t = LinRange{T}(kn[1], kn[end], n+1)
    h = step(t)

    y = exp.(eval_linpred(shs, params, t)) # (length(t), length(samples)) matrix
    l1_norms = Vector{T}(undef, length(params))
    for i in eachindex(params)
        l1_norms[i] = h / 3 * (y[1, i] + 4*sum(y[2:2:n, i]) + 2 * sum(y[3:2:n-1, i]) + y[n+1, i])
    end
    return l1_norms
end

function eval_linpred(shs::SHSModel{T, A, D}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}) where {T<:Real, A, D, Names, Vals, S<:Real}
    Z = demmler_reinsch_basis_matrix(t, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t)), t, Z)

    # Compute linear predictor
    linpreds = C * params.β
    return linpreds
end
function eval_linpred(shs::SHSModel{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}) where {T<:Real, A, D, Names, Vals, S<:Real}
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

# Compute linear binning on a regular grid
function linear_binning(x::AbstractVector{T}, n_bins::Int, xmin::T, xmax::T) where {T<:Real}
    N = zeros(T, n_bins)
    delta = (xmax - xmin) / n_bins

    for i = eachindex(x)
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