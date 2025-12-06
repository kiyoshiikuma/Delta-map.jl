# using Original Delta-map method to calculate likelihood function and Clean CMB map
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
using SparseArrays
using SymPy
@pyimport healpy as hp
@pyimport numpy as np

mutable struct SetParams
    freq_bands::Vector{Int}          # GHz
    which_model::String              # (e.g.) "s1", "d0", "d1", "d1 and s1"
    r_input::Float64
    cmb_freq::Int                    # GHz
    seed::Int
    nside::Int
    #cov_mat_scal::Matrix{Float64}    # (2*npix)×(2*npix)
    #cov_mat_tens::Matrix{Float64}    # (2*npix)×(2*npix)
    mask::Vector{Float64}            # length=npix, 1/0
    Q_map::Vector{Vector{Float64}}   # each length=npix
    U_map::Vector{Vector{Float64}}   # each length=npix
    N_set::Vector{Matrix{Float64}}  # (unused here, keep for compatibility)
    N⁻¹_set::Vector{Matrix{Float64}} # (unused here, keep for compatibility)
    """Parameters for r estimation"""
end

mutable struct FitParams
    beta_s::Float64
    beta_d::Float64
    T_d::Float64
    r_est::Float64
    """Optimization Parameters"""
end

function _mask_idx(set_params::SetParams)
    return findall(x -> x != 0, set_params.mask)  # pixel indices (1..npix)
end

function extract_masked_values(set_params::SetParams, map_vec::AbstractVector)
    idx = _mask_idx(set_params)
    return map_vec[idx]
end

function extract_masked_elements(set_params::SetParams, A::AbstractMatrix)
    """
    Extract masked submatrix from a (2*npix)×(2*npix) matrix for [Q;U].
    output: (2*nmask)×(2*nmask)
    """
    idx = _mask_idx(set_params)
    npix = length(set_params.mask)
    idxQU = vcat(idx, idx .+ npix)
    return A[idxQU, idxQU]
end

# =========================
# Covariance + logdet
# =========================

function calc_all_cov_mat(set_params::SetParams, fit_params::FitParams)
    """C = C_scal + r*C_tens"""
    return set_params.cov_mat_scal .+ fit_params.r_est .* set_params.cov_mat_tens
end

function cholesky_decomposition(A::AbstractMatrix)
    """Cholesky for SPD matrix (symmetrize for safety)"""
    n, m = size(A)
    n == m || throw(ArgumentError("Matrix must be square."))
    As = Symmetric((A + A') / 2)
    isposdef(As) || throw(ArgumentError("Matrix is not positive definite."))
    return cholesky(As).L
end

function positive_definite_inverse(A::AbstractMatrix)
    """
    Compute the inverse of a positive definite matrix using Cholesky decomposition
    and solving with identity matrix.
    """
    # Perform Cholesky decomposition
    A_cho = cholesky(A)
    # Create an identity matrix
    uni = Matrix(I, size(A))
    # Solve for the inverse using the decomposed matrix
    A_inv = A_cho \ uni
    return A_inv
end

function cholesky_logdet(A::AbstractMatrix)
    """logdet(A) for SPD A via Cholesky"""
    L = cholesky_decomposition(A)
    return 2 * sum(log.(diag(L)))
end

# =========================
# Foreground response vectors
# =========================

function calc_D_elements(fit_params::FitParams, freq)
    # Define symbols used in the calculations
    ν, ν_s_star, ν_d_star, β_s, β_d, T_d, h, k_B = symbols("ν ν_s_star ν_d_star β_s β_d T_d h k_B")
    # Definitions of x_d and related variables
    x_d = (ν / T_d) / (2.083661912 * 10^10)
    x_d_star = (ν_d_star / T_d) / (2.083661912 * 10^10)
    x_cmb = (ν / 2.725) / (2.083661912 * 10^10)
    g = ((exp(x_cmb) - 1)^2) / (exp(x_cmb) * x_cmb^2)
    # Equations for D^d_ν and D^s_ν
    D_ν_d = g * (ν / ν_d_star)^(β_d + 1) * (exp(x_d_star) - 1) / (exp(x_d) - 1)
    D_ν_s = g * (ν / ν_s_star)^(β_s)
    # Partial derivatives
    partial_beta_s_1st = diff(D_ν_s, β_s)
    partial_beta_s_2nd = diff(partial_beta_s_1st, β_s)
    partial_beta_d_1st = diff(D_ν_d, β_d)
    partial_beta_d_2nd = diff(partial_beta_d_1st, β_d)
    partial_T_d_1st = diff(D_ν_d, T_d)
    partial_T_d_2nd = diff(partial_T_d_1st, T_d)
    partial_beta_d_T_d = diff(partial_beta_d_1st, T_d)
    # Substitute values into symbols
    subs_values = Dict(
        ν => freq * 10^9,
        ν_s_star => 23 * 10^9,
        ν_d_star => 353 * 10^9,
        β_s => fit_params.beta_s,
        β_d => fit_params.beta_d,
        T_d => fit_params.T_d
    )
    # Evaluate results
    s = N(subs(D_ν_s, subs_values))
    ss = N(subs(partial_beta_s_1st, subs_values))
    sss = N(subs(partial_beta_s_2nd, subs_values))
    d = N(subs(D_ν_d, subs_values))
    dd = N(subs(partial_beta_d_1st, subs_values))
    ddd = N(subs(partial_T_d_1st, subs_values))
    dddd = N(subs(partial_beta_d_2nd, subs_values))
    ddddd = N(subs(partial_T_d_2nd, subs_values))
    dddddd = N(subs(partial_beta_d_T_d, subs_values))
    # Return all calculated values
    return s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd
end

function calc_d_s_vec(fit_params::FitParams, freq::Int)
    s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd = calc_D_elements(fit_params, freq) 
    #return [s], [d]
    return [s, ss], [d, dd, ddd]
    #return [s, ss, sss], [d, dd, ddd, dddd, ddddd, dddddd]
end

# =========================
# A matrix (constraints)
# =========================

function calc_A(set_params::SetParams, fit_params::FitParams)
    """
    A has columns for non-CMB channels in the order of freq_bands (excluding cmb_freq).
    For each freq, column is d_vec or s_vec depending on model.
    """
    A_s = Vector{Vector{Float64}}()
    A_d = Vector{Vector{Float64}}()

    for freq in set_params.freq_bands
        freq == set_params.cmb_freq && continue

        s_vec, d_vec = calc_d_s_vec(fit_params, freq)

        if set_params.which_model == "s1"
            push!(A_s, s_vec)
        elseif set_params.which_model == "d0" || set_params.which_model == "d1"
            push!(A_d, d_vec)
        elseif set_params.which_model == "d1 and s1"
            push!(A_d, d_vec)
            push!(A_s, s_vec)
        else
            error("Invalid model: $(set_params.which_model). Use 's1', 'd1', or 'd1 and s1'.")
        end
    end

    if set_params.which_model == "s1"
        return hcat(A_s...)
    elseif set_params.which_model == "d0" || set_params.which_model == "d1"
        return hcat(A_d...)
        elseif set_params.which_model == "d1 and s1"
        Ad = hcat(A_d...)
        As = hcat(A_s...)
        return vcat(Ad, As)
    end
end

# =========================
# Q-min (constraint + quadratic minimization)
# =========================

noncmb_freqs(sp::SetParams) = [f for f in sp.freq_bands if f != sp.cmb_freq]

function vec_cmb(sp::SetParams, fp::FitParams)
    s_cmb, d_cmb = calc_d_s_vec(fp, sp.cmb_freq)
    if sp.which_model == "s1"
        return s_cmb
    elseif sp.which_model == "d0" || sp.which_model == "d1"
        return d_cmb
    elseif sp.which_model == "d1 and s1"
        return vcat(d_cmb, s_cmb)
    else
        error("Invalid model: $(sp.which_model). Use 's1', 'd0', 'd1', or 'd1 and s1'.")
    end
end

function calc_alpha_Qmin(sp::SetParams, fp::FitParams, Q::AbstractMatrix)
    """
    Solve: min α'Qα  s.t.  Aα = -d_cmb
    α* = Q^{-1}A'(AQ^{-1}A')^{-1}(-d_cmb)
    """
    A = calc_A(sp, fp)        # (n_param × n_nonCMB)
    b = -vec_cmb(sp, fp)      # (n_param)

    n_non = length(noncmb_freqs(sp))
    (size(Q,1) == n_non && size(Q,2) == n_non) ||
        error("Q size mismatch: expected $(n_non)×$(n_non), got $(size(Q)).")

    X = Q \ transpose(A)      # Q^{-1}A'  (n_nonCMB × n_param)
    M = A * X                 # AQ^{-1}A' (n_param × n_param)
    y = M \ b
    alpha_hat = X * y         # (n_nonCMB)

    # insert CMB coefficient = 1
    freqs = collect(sp.freq_bands)
    cmb_i = findfirst(==(sp.cmb_freq), freqs)
    cmb_i === nothing && error("cmb_freq=$(sp.cmb_freq) is not in freq_bands.")

    alpha_full = collect(alpha_hat)
    insert!(alpha_full, cmb_i, 1.0)

    denom = 1.0 + sum(alpha_hat)
    return alpha_full, denom
end

# =========================
# Clean map (compatible interface)
# =========================

function calc_Clean_map(set_params::SetParams, fit_params::FitParams; Qweight::Union{Nothing,AbstractMatrix}=nothing)
    """
    If Qweight === nothing:
        solve A*α = -d_cmb (as before)
    If Qweight given:
        solve Q-min constrained solution
    """
    freqs = collect(set_params.freq_bands)
    cmb_i = findfirst(==(set_params.cmb_freq), freqs)
    cmb_i === nothing && error("cmb_freq=$(set_params.cmb_freq) is not in freq_bands.")

    if Qweight === nothing
        @info "Qweight is nothing: solving A*α = -d_cmb (standard solution) to estimate α."
        A_mat = calc_A(set_params, fit_params)
        println("A_mat size: ", size(A_mat))
        v = vec_cmb(set_params, fit_params)
        alpha_hat = - A_mat \ v

        alpha_full = collect(alpha_hat)
        insert!(alpha_full, cmb_i, 1.0)
        denom = 1.0 + sum(alpha_hat)
    else
        @info "Qweight is provided: computing the Q-min constrained solution to estimate α."
        alpha_full, denom = calc_alpha_Qmin(set_params, fit_params, Qweight)
    end

    Q = zero(set_params.Q_map[1])
    U = zero(set_params.U_map[1])
    for i in eachindex(set_params.freq_bands)
        Q .+= alpha_full[i] .* set_params.Q_map[i]
        U .+= alpha_full[i] .* set_params.U_map[i]
    end
    Q ./= denom
    U ./= denom
    x = vcat(Q, U)

    return Q, U, x, alpha_full
end

# =========================
# chi^2 and likelihood
# =========================

function calc_chi_sq(set_params::SetParams, fit_params::FitParams; Qweight::Union{Nothing,AbstractMatrix}=nothing)
    pol_sen = 0.2  # μK

    # noise realization (seed controls the map)
    Random.seed!(set_params.seed)
    art_noise_map, _ = calc_noise_map(pol_sen, set_params.nside)

    # noise covariance (no seed dependency)
    art_noise_cov_mat = calc_noise_cov_mat(pol_sen, set_params.nside)

    # masked covariance
    cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) .+ art_noise_cov_mat)

    # clean map
    Q_1, U_1, x_map, alpha_i = calc_Clean_map(set_params, fit_params; Qweight=Qweight)

    # build IQU(3,npix) for smoothing
    npix = hp.nside2npix(set_params.nside)
    iqu = zeros(3, npix)
    iqu[2, :] .= Q_1
    iqu[3, :] .= U_1

    # your original smoothing rule
    fwhm_con = 2200 * (4 / set_params.nside)^2
    sm = smoothing_map_fwhm(iqu, fwhm_con, set_params.nside)

    # add noise to Q/U AFTER smoothing (as your original)
    smQ = Vector(sm[2, :]) .+ art_noise_map.Q
    smU = Vector(sm[3, :]) .+ art_noise_map.U

    xQ_masked = extract_masked_values(set_params, smQ)
    xU_masked = extract_masked_values(set_params, smU)
    x_sm_masked = vcat(xQ_masked, xU_masked)

    # chi^2 = x' C^{-1} x (use / form like your original)
    return x_sm_masked' / cov_mat * x_sm_masked
end

function calc_likelihood(set_params::SetParams, fit_params::FitParams; Qweight::Union{Nothing,AbstractMatrix}=nothing)
    pol_sen = 0.2
    art_noise_cov_mat = calc_noise_cov_mat(pol_sen, set_params.nside)

    cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) .+ art_noise_cov_mat)
    chi_sq = calc_chi_sq(set_params, fit_params; Qweight=Qweight)
    det_C = cholesky_logdet(cov_mat)

    return chi_sq + det_C
end

# =========================
# Set data model
# =========================
function set_truncate_N_N⁻¹!(set_params::SetParams, lmin, lmax; tag::AbstractString)
    """
    Set N⁻¹
    """
    N_set = []
    N⁻¹_set = []
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    for nu_i in set_params.freq_bands
        freq_name = string(nu_i)
        dir_noise = "/Users/ikumakiyoshi/Library/Mobile Documents/com~apple~CloudDocs/study_fg_rm/program/julia_Delta_map/Delta_map/make_noise_covariance_matrix/smoothing_noise_cov_mat/"
        nside_name = "nside_"
        nside_n = string(set_params.nside)
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name,
                                lmin_n, lmin, lmax_n, lmax,
                                "_truncate_noise_cov_mat_with_smoothing_", tag, ".npy")
        noise_cov_mat = npzread(noise_cov_name)
        noise_cov_inv = positive_definite_inverse(
            extract_masked_elements(set_params, noise_cov_mat + art_noise_cov_mat)
        )
        push!(N_set, noise_cov_mat)
        push!(N⁻¹_set, noise_cov_inv)
    end
    set_params.N_set = N_set
    set_params.N⁻¹_set = N⁻¹_set
end

# =========================
# Q weight matrix
# =========================
function make_Qweight_sigma2(set_params::SetParams, noise_cov_mats::AbstractDict, bands)
    idx   = findall(!iszero, set_params.mask)
    nmask = length(idx)

    Qw = zeros(length(bands), length(bands))
    for (i, nu) in enumerate(bands)
        N = noise_cov_mats[nu]  # (2nmask, 2nmask) を想定
        NQQ = @view N[1:nmask, 1:nmask]
        NUU = @view N[nmask+1:end, nmask+1:end]
        Qw[i,i] = (tr(NQQ) + tr(NUU)) / (2*nmask)
    end
    return Qw
end

function make_Qweight_sigma2(set_params::SetParams; include_art_noise::Bool=true, pol_sen::Float64=0.2)
    length(set_params.N_set) == length(set_params.freq_bands) || error("N_set length mismatch")

    bands = filter(nu -> nu != set_params.cmb_freq, set_params.freq_bands)

    noise_cov_mats = Dict(
        nu => extract_masked_elements(set_params, set_params.N_set[i])
        for (i, nu) in enumerate(set_params.freq_bands) if nu != set_params.cmb_freq
    )

    Qw = make_Qweight_sigma2(set_params, noise_cov_mats, bands)

    if include_art_noise
        artN = calc_noise_cov_mat(pol_sen, set_params.nside)
        Nart = extract_masked_elements(set_params, artN)
        idx   = findall(!iszero, set_params.mask)
        nmask = length(idx)
        NQQ = @view Nart[1:nmask, 1:nmask]
        NUU = @view Nart[nmask+1:end, nmask+1:end]
        sig2_art = (tr(NQQ) + tr(NUU)) / (2*nmask)
        return Qw + sig2_art * I
    else
        return Qw
    end
end


function make_simple_Qweight_sigma2(set_params::SetParams; include_art_noise::Bool=true, pol_sen::Float64=0.2)
    bands = filter(nu -> nu != set_params.cmb_freq, set_params.freq_bands)

    Qw = zeros(length(bands), length(bands))
    for (i, nu_i) in enumerate(bands)
        _, sigma, _, _ = truncate_noise_sigma_calc(nu_i, set_params.nside)
        Qw[i, i] = sigma^2
    end
    return Qw  # ←CMB成分は最初から入らない（サイズも1つ小さくなる）
end
