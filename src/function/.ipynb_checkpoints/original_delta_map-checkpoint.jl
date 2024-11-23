# using Original Delta-map method to calculate likelihood function and Clean CMB map
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
@pyimport healpy as hp

mutable struct SetParams
    freq_bands::Vector{Int}
    which_model::String
    r_input::Float64
    cmb_freq::Int
    seed::Int
    nside::Int
    cov_mat_scal::Matrix{Float64}
    cov_mat_tens::Matrix{Float64}
    mask::Vector{Float64} 
    Q_map::Vector{Vector{Float64}}
    U_map::Vector{Vector{Float64}} 
    N⁻¹_set::Vector{Matrix{Float64}}
    """Parameters for r estimation"""
end

mutable struct FitParams
    beta_s::Float64
    beta_d::Float64
    T_d::Float64
    r_est::Float64
    """Optimization Parameters"""
end

function smoothing_map_fwhm(input_map, input_fwhm, nside)
    """Smoothing map with input_fwhm"""
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    smoothed_map = hp.sphtfunc.alm2map(alm, nside, lmax=2*nside, pixwin=true, verbose=false, fwhm=input_fwhm*pi/10800.)
    return smoothed_map
end

function extract_masked_values(set_params::SetParams, map)
    return [map[i] for i in 1:length(set_params.mask) if set_params.mask[i] == 1]
end

function extract_masked_elements(set_params::SetParams, A::AbstractMatrix)
    # Read the mask
    nonzero_indices = findall(x -> x != 0, set_params.mask)
    n = length(nonzero_indices)
    masked_elements = zeros(2n, 2n)
    m_size = Int(size(A)[1] / 2)
    # Extract masked elements
    for (i, non_mask_i) in enumerate(nonzero_indices)
        for (j, non_mask_j) in enumerate(nonzero_indices)
            masked_elements[i, j] = A[non_mask_i, non_mask_j]
            masked_elements[i + n, j] = A[non_mask_i + m_size, non_mask_j]
            masked_elements[i, j + n] = A[non_mask_i, non_mask_j + m_size]
            masked_elements[i + n, j + n] = A[non_mask_i + m_size, non_mask_j + m_size]
        end
    end
    return masked_elements
end

function calc_all_cov_mat(set_params::SetParams, fit_params::FitParams)
    """
    Calculate sum of covariance matrix.
    """
    cov_mat = set_params.cov_mat_scal + fit_params.r_est * set_params.cov_mat_tens 
    return cov_mat
end

function positive_definite_inverse(A::AbstractMatrix)
    """
    Compute the inverse of a positive definite matrix.
    """
    # Check if the matrix is symmetric
    if A != A'
        throw(ArgumentError("Matrix is not symmetric."))
    end
    # Check if the matrix is positive definite
    if !isposdef(A)
        throw(ArgumentError("Matrix is not positive definite."))
    end
    # Perform Cholesky decomposition
    A_cho = cholesky(A)
    # Get the lower triangular matrix
    A_cho_L = A_cho.L
    # Create an identity matrix
    uni = Matrix{eltype(A)}(I, size(A))
    # Compute the inverse of the lower triangular matrix (precision-focused)
    A_cho_L_inv = uni / A_cho_L
    # Compute the inverse of the matrix
    A_inv = A_cho_L_inv' * A_cho_L_inv
    # Convert the upper or lower triangular matrix to a symmetric matrix
    A_inv_U = Symmetric(A_inv, :U)
    A_inv_L = Symmetric(A_inv, :L)
    A_true = 0.5 * (A_inv_U .+ A_inv_L)
    return A_true
end

function cholesky_decomposition(A::AbstractMatrix)
    """
    Perform Cholesky decomposition on a positive definite matrix.
    """
    # Default tolerance
    default_tol = 1e-11
    # Check if the matrix is square
    n, m = size(A)
    if n != m
        throw(ArgumentError("Matrix must be square."))
    end
    # Symmetrize the matrix
    A = (A + A') / 2
    # Check if the matrix is positive definite
    if !isposdef(A)
        throw(ArgumentError("Matrix is not positive definite."))
    end
    # Perform Cholesky decomposition and get the lower triangular matrix
    A_cho = cholesky(A).L
    return A_cho
end

function cholesky_logdet(A::AbstractMatrix)
    """
    Compute the log-determinant of a positive definite matrix using Cholesky decomposition.
    """
    # Perform Cholesky decomposition
    A_cho = cholesky_decomposition(A)
    # Twice the log-determinant of the matrix
    logdet = 2 * sum(log.(diag(A_cho)))
    return logdet
end

function calc_D_elements(fit_params::FitParams, freq::Int)
    """Calculate D elements"""
    freq_bs = 23 * 10^9
    freq_bd = 353 * 10^9
    x = (freq / 2.725) / (2.083661912 * 10^10)
    g_freq = ((exp(x) - 1)^2) / (exp(x) * x^2)
    s = g_freq * (freq / freq_bs)^(fit_params.beta_s)
    ss = g_freq * (freq / freq_bs)^(fit_params.beta_s) * log(freq / freq_bs)
    x_d = (freq / fit_params.T_d) / (2.083661912 * 10^10)
    x_bd = (freq_bd / fit_params.T_d) / (2.083661912 * 10^10)
    d = g_freq * (freq / freq_bd)^(fit_params.beta_d + 1) * ((exp(x_bd) - 1) / (exp(x_d) - 1))
    dd = d * log(freq / freq_bd)
    ddd = d * (((x_d * exp(x_d)) / (exp(x_d) - 1)) - (x_bd * exp(x_bd)) / (exp(x_bd) - 1)) / fit_params.T_d
    return s, ss, d, dd, ddd
end

function calc_d_s_vec(fit_params::FitParams, freq::Int)
    s, ss, d, dd, ddd = calc_D_elements(fit_params::FitParams, freq * 10^9)
    s_vec = [s, ss]
    d_vec = [d, dd, ddd]
    return s_vec, d_vec
end

function calc_A(set_params::SetParams, fit_params::FitParams)
    A_s = []
    A_d = []
    for freq in set_params.freq_bands
        s_vec, d_vec = calc_d_s_vec(fit_params, freq)
        if cmb_freq != freq
            if which_model == "s1"
                push!(A_s, s_vec)
            elseif which_model == "d1" || set_params.which_model == "d0"
                push!(A_d, d_vec)
            elseif which_model == "d1 and s1"
                push!(A_s, s_vec)
                push!(A_d, d_vec)
            end
        end
    end
    if set_params.which_model == "s1"
        A = hcat(A_s...)
    elseif set_params.which_model == "d1" || which_model == "d0"
        A = hcat(A_d...)
    elseif set_params.which_model == "d1 and s1"
        A_d = hcat(A_d...)
        A_s = hcat(A_s...)
        A = vcat(A_d, A_s)
    else
        error("Invalid model specified. Please enter a valid model: 's1', 'd1', 'd0', or 'd1 and s1'.")
    end
    return A
end

function calc_Clean_map(set_params::SetParams, fit_params::FitParams)
    A_mat = calc_A(set_params, fit_params)
    s_cmb, d_cmb = calc_d_s_vec(fit_params, set_params.cmb_freq)
    if set_params.which_model == "s1"
        vec_cmb = s_cmb
    elseif set_params.which_model == "d1" || set_params.which_model == "d0"
        vec_cmb = d_cmb
    elseif set_params.which_model == "d1 and s1"
        vec_cmb = vcat(d_cmb, s_cmb)
    end
    alpha_i = -A_mat \ vec_cmb
    Q = zeros(size(set_params.Q_map[1]))
    U = zeros(size(set_params.U_map[1]))
    sum_alpha = sum(alpha_i)
    freq_list = collect(set_params.freq_bands)
    cmb_index = findall(x -> x == set_params.cmb_freq, freq_list)[1]
    insert!(alpha_i, cmb_index, 1.0)
    for (i, freq) in enumerate(set_params.freq_bands)
        Q .+= alpha_i[i] * set_params.Q_map[i]
        U .+= alpha_i[i] * set_params.U_map[i]
    end
    Q /= 1 + sum_alpha
    U /= 1 + sum_alpha
    x = vcat(Q, U)
    return Q, U, x, alpha_i
end

function calc_chi_sq(set_params::SetParams, fit_params::FitParams)
    noise_seed = [1, 2, 3]
    pol_sen = 0.2 # μK  
    art_noise_map = calc_noise_map(pol_sen, set_params.nside, [1, 1, 1])
    art_noise_map, sigma = calc_noise_map(pol_sen, nside, noise_seed);
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    # Calculate the covariance matrix
    cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat)
    # Calculate the clean map
    Q_1, U_1, x_map, alpha_i = calc_Clean_map(set_params, fit_params)
    npix = hp.nside2npix(set_params.nside)
    xx_map = [x_map[1:npix]'.*0; x_map[1:npix]'; x_map[npix + 1:2 * npix]']
    smootinhg_x_map = smoothing_map_fwhm(xx_map, 2200, set_params.nside) 
    smoosthing_masked_x_map_Q = extract_masked_values(set_params, smootinhg_x_map[2,:] + art_noise_map[1])
    smoosthing_masked_x_map_U = extract_masked_values(set_params, smootinhg_x_map[3,:] + art_noise_map[2])
    x_sm_msked = [smoosthing_masked_x_map_Q ; smoosthing_masked_x_map_U]
    return x_sm_msked' / cov_mat * x_sm_msked
end

function calc_likelihood(set_params::SetParams, fit_params::FitParams)
    noise_seed = [1, 2, 3]
    pol_sen = 0.2 # μK  
    art_noise_map, sigma = calc_noise_map(pol_sen, nside, noise_seed);
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    # Calculate the covariance matrix
    cov_mat = extract_masked_elements(set_params,  calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat)
    chi_sq = calc_chi_sq(set_params, fit_params)
    det_C = cholesky_logdet(cov_mat)
    chi_sq + det_C
end