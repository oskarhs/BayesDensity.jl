# Find the mixture weights corresponding to given coefficients in the unnormalized B-spline basis
function coef_to_theta(coef::AbstractVector{T}, basis::AbstractBSplineBasis) where {T<:Real}
    θ = coef ./ compute_norm_fac(basis)
    return θ
end

# Find the B-spline coefficients corresponding to given mixture weights in the normalized B-spline basis
function theta_to_coef(θ::AbstractVector{T}, basis::AbstractBSplineBasis) where {T<:Real}
    coef = θ .* compute_norm_fac(basis, T)
    return coef
end

# Compute the vector Z of normalizing constants for a given B-spline basis on [0,1].
# The resulting normalized B-spline basis is given by bₖ(x) = Bₖ(x) / Zₖ
function compute_norm_fac(basis::AbstractBSplineBasis{N, S}, T::Type{<:Real}=Float64) where {N, S<:Real}
    K = length(basis)
    norm_fac = Vector{S}(undef, K)
    bmin, bmax = boundaries(basis)
    for k in 1:K
        F = integral(Spline(basis, unitvector(K, k, T)))
        norm_fac[k] = 1/(F(bmax) - F(bmin))
    end
    return norm_fac
end

# Compute the vector μ such that ∑ₖ θₖ bₖ(x) = 1 for all x, where θ = stickbreaking(μ)
function compute_μ(basis::AbstractBSplineBasis{N, S}) where {N, S<:Real}
    xmin, xmax = boundaries(basis)
    K = length(basis)
    p0 = coef_to_theta(fill(S(1/(xmax-xmin)), K), basis)

    μ = Vector{S}(undef, K-1)
    θ_cum = Vector{S}(undef, K)
    θ_cum[1] = 0

    for k in 1:K-1
        μ[k] = logit(p0[k] / (1-θ_cum[k]))
        θ_cum[k+1] = θ_cum[k] + p0[k]
    end
    return μ
end

#NB! Upate to take spline basis as argument
function create_spline_basis_matrix(x::AbstractVector{T}, basis::AbstractBSplineBasis{N, S}) where {T<:Real, N, S<:Real}
    K = length(basis)
    R = promote_type(T, S)

    n = length(x)
    b_ind = Vector{Int}(undef, n)
    B = Matrix{R}(undef, (n, 4))
    norm_fac = compute_norm_fac(basis, R)
    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in eachindex(x)
        j, basis_eval = basis(x[i])
        b_ind[i] = j-3 # So we compute b_{j-3}, b_{j-2}, b_{j-1} and b_j for x_i
        B[i,:] .= reverse(basis_eval) .* norm_fac[b_ind[i]:b_ind[i]+3]
    end
    return B, b_ind
end

function create_spline_basis_matrix_binned(x::AbstractVector{T}, basis::AbstractBSplineBasis{N, S}, n_bins::Integer) where {T<:Real, N, S<:Real}
    R = promote_type(T, S)
    K = length(basis)
    deg = order(basis) - 1

    bounds = boundaries(basis)

    n_bins = (fld(n_bins, K-2)+1)*(K-2)-1 # Make the number of bins a multiple of K-2 so that at most 4 basis functions are nonzero at a time
    bincounts = bin_regular(x, bounds[1], bounds[2], n_bins, true)
    binedges = LinRange(bounds[1], bounds[2], n_bins+1)
    n = length(x)
    b_ind = Vector{Int}(undef, n_bins)
    B = Matrix{R}(undef, (n_bins, 4))
    norm_fac = compute_norm_fac(basis, R)
    
    # Compute ∫ bⱼ(x) dx over each bin for the nonzero coefficients

    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    basis_knots = unique(knots(basis))
    for i in 1:n_bins
        x0 = binedges[i]
        x1 = binedges[i+1]
        j = find_knot_interval(basis_knots, x0)[1] # So we compute b_{j-3}, b_{j-2}, b_{j-1} and b_j for x_i
        b_ind[i] = j
        for l in 1:4
            k = j + l - 1
            F = integral(Spline(basis, unitvector(K, k, T)))
            B[i,l] = (F(x1) - F(x0)) * norm_fac[k]
        end
    end
    return B, b_ind, bincounts
end


function create_unnormalized_sparse_spline_basis_matrix(x::AbstractVector{T}, basis::AbstractBSplineBasis{N, S}) where {T<:Real, N, S<:Real}
    K = length(basis)

    n = length(x)

    I = Vector{Int}(undef, N*n) # row indices
    J = Vector{Int}(undef, N*n) # column indices
    V = Vector{promote_type(T, S)}(undef, N*n)
    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in eachindex(x)
        ind = (N*i-N+1):(N*i)
        j, basis_eval = basis(x[i])
        I[ind] .= i
        j = max(N, j)
        J[ind] .= (j-N+1):j
        V[ind] .= reverse(basis_eval)
    end
    return sparse(I, J, V, n, K)
end

# Get the B-spline basis corresponding to the integral of a B-spline
# NB! BSplineKit has a function that does this but it is not exported/declared public API, so we maintain our own version here.
function integrate_spline_basis(basis::BSplineBasis)
    kn = knots(basis)
    m = order(basis)
    
    # Create new spline knots (same internal knots as previously, with 2 new spline knots around the edges.)
    kn_int = similar(kn, length(kn) + 2)
    kn_int[(begin + 1):(end - 1)] .= kn
    kn_int[begin] = kn_int[begin + 1]
    kn_int[end] = kn_int[end - 1]
    return BSplineBasis(BSplineOrder(m + 1), kn_int; augment = Val(false))
end

# Create basis matrix of the B-Spline basis corresponding to the integrated B-spline
function _create_integrated_sparse_spline_basis_matrix(x::AbstractVector{<:Real}, basis::AbstractBSplineBasis)
    basis_int = integrate_spline_basis(basis)
    B_int = create_unnormalized_sparse_spline_basis_matrix(x, basis_int)
    return B_int
end

# Get the coefficients of ∫ ∑ₖ θₖ bₖ(s) ds in terms of the integrated spline basis.
function _get_integrated_spline_coefs(basis::AbstractBSplineBasis{N, T}, spline_coefs::AbstractMatrix{S}) where {N, T, S}
    kn = knots(basis)
    K = length(basis)
    R = promote_type(T, S)

    # Get the dimension of the integrated spline coefficients
    new_dims = size(spline_coefs, 1)+1, size(spline_coefs, 2)
    spline_coefs_int = similar(spline_coefs, R, new_dims)
    for i in axes(spline_coefs_int, 2)
        spline_coefs_int[1,i] = zero(R)
        spline_coefs_int[2:end,i] = cumsum(spline_coefs[:,i] .* (view(kn, N+1:N+K) - view(kn, 1:K)) / N)
    end

    return spline_coefs_int
end

function _get_integrated_spline_coefs(basis::AbstractBSplineBasis{N, T}, spline_coefs::AbstractVector{S}) where {N, T, S}
    kn = knots(basis)
    K = length(basis)
    R = promote_type(T, S)

    # Get the dimension of the integrated spline coefficients
    spline_coefs_int = similar(spline_coefs, R, length(spline_coefs)+1)
    spline_coefs_int[1] = zero(R)
    spline_coefs_int[2:end] = cumsum(spline_coefs .* (view(kn, N+1:N+K) - view(kn, 1:K)) / N)

    return spline_coefs_int
end


# KL-divergence between two multivariate normal distributions in their canonical form (takes potential, covmatrix as inputs)
# Note: The implementation in Distributions.jl forms the covariance matrix itself, which is dense.
# Here, we use sparse solver methods instead for efficiency and stability.
function Distributions.kldivergence(h1, Q1, h2, Q2)
    μ1 = Q1 \ h1
    μ2 = Q2 \ h2
    return 1/2 * (dot(μ2 - μ1, Q2, μ1 - μ2) + tr(Q1 \ Q2) - logabsdet(Q2) + logabsdet(Q1) - length(h1))
end