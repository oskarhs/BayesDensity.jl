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
        #V[ind] .= reverse(basis_eval) .* norm_fac[(j-3):j]
        V[ind] .= reverse(basis_eval)
    end
    return sparse(I, J, V, n, K)
end