using LinearAlgebra, BandedMatrices, SelectedInversion

function varinf(bsm::BSMModel; init_params=get_default_initparams(bsm), max_iter::Int=500) # Also: tolerance parameters
    return _variational_inference(bsm, init_params, max_iter)
end

# Can do something more sophisticated here at a later point in time if we get a good idea.
get_default_initparams(bsm::BSMModel{T, A, NT}) where {T, A, NT} = compute_μ(basis(bsm), T)

function _sample_posterior(rng::AbstractRNG, bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int) where {T, A, Vals, NT}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)
end

#= K = 200
P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))
Q = transpose(P) * P + Diagonal(ones(199))
inverse_diagonals_penta(Q[band(0)], Q[band(1)], Q[band(2)])

P = spdiagm(K-3, K-1, 0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3))
Q = transpose(P) * P + Diagonal(ones(199))

begin
    Z, _ = selinv(Q; depermute=true)
    d0 = Vector(diag(Z))
    d1 = Vector(diag(Z, 1))
    d2 = Vector(diag(Z, 2))
end
 =#