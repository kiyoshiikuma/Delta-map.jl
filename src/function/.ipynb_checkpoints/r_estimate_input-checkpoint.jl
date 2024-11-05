using NPZ
using PyCall
using LinearAlgebra
include("calc_noise.jl")
@pyimport healpy as hp
@pyimport iminuit as iminuit
@pyimport numpy as np

mutable struct SetParams
    r_input::Float64
    seed::Int
    nside::Int
    cov_mat_scal::Matrix{Float64}
    cov_mat_tens::Matrix{Float64}
    mask::Vector{Float64} 
    Q_map::Vector{Float64} 
    U_map::Vector{Float64} 
    N⁻¹_set::Vector{Matrix{Float64}}
    """Parameters for r estimation"""
end

mutable struct FitParams
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

function calc_likelihood_for_input(set_params::SetParams, fit_params::FitParams)
    noise_seed = [1, 2, 3]
    pol_sen = 0.2 # μK  
    art_noise_map = calc_noise_map(pol_sen, set_params.nside, [1, 1, 1])
    art_noise_map, sigma = calc_noise_map(pol_sen, nside, noise_seed);
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    # Calculate the covariance matrix
    cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat)
    # load_input_map
    x_map = vcat(set_params.Q_map, set_params.U_map)
    npix = hp.nside2npix(set_params.nside)
    xx_map = [x_map[1:npix]'.*0; x_map[1:npix]'; x_map[npix + 1:2 * npix]']
    smootinhg_x_map = smoothing_map_fwhm(xx_map, 2200, set_params.nside) 
    smoosthing_masked_x_map_Q = extract_masked_values(set_params, smootinhg_x_map[2,:] + art_noise_map[1])
    smoosthing_masked_x_map_U = extract_masked_values(set_params, smootinhg_x_map[3,:] + art_noise_map[2])
    x_sm_msked = [smoosthing_masked_x_map_Q ; smoosthing_masked_x_map_U]
    det_C = cholesky_logdet(cov_mat)
    return x_sm_msked' / cov_mat * x_sm_msked + det_C
end

function iterative_minimization(set_params::SetParams, fit_params::FitParams)
    # initialize parameters
    r_pre = Inf
    r_out = 0.5
    neg2L_pre = -Inf
    neg2L_out = 1e10
    # loop until convergence
    while !(abs(neg2L_pre - neg2L_out) <= 1e-2 && abs(r_pre - r_out) <= 1e-5)
        # minimize likelihood_wrapper for r
        function likelihood_wrapper(r::Vararg{Float64})
            fit_params.r_est = r[1]
            return calc_likelihood_for_input(set_params, fit_params)
        end
        minuit_r = iminuit.Minuit(likelihood_wrapper, fit_params.r_est)
        minuit_r[:limits] = [(0.0, Inf)]
        minuit_r[:errordef] = 1
        minuit_r.migrad()
        r_out = minuit_r[:values][1]
        # reset fg parameters
        r_pre = fit_params.r_est
        fit_params.r_est = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood_for_input(set_params, fit_params)
    end
    # return the fitted r value
    return fit_params.r_est
end

function calc_likelihood_for_input_add_noise(set_params::SetParams, fit_params::FitParams, freq::Int)
    noise_seed = [1, 2, 3]
    pol_sen = 0.2 # μK  
    art_noise_map, sigma = calc_noise_map(pol_sen, set_params.nside, noise_seed);
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    # Calculate the noise map
    noise_map, sigma = noise_sigma_calc(freq, set_params.nside, [3, 4, 5] .+ 20 * set_params.seed)
    # load noise covariance matrix
    freq_name = string(freq)
    dir_noise = "/Users/ikumakiyoshi/Library/Mobile Documents/com~apple~CloudDocs/study_fg_rm/program/Deltamap_test/julia_delta-map/make_map/noise_map_make/smoothing_noise_cov_mat/"
    nside_name = "nside_"
    nside_n = string(set_params.nside)
    # Load noise cov_mat
    noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, "_noise_cov_mat_with_smoothing.npy")
    noise_cov_mat = npzread(noise_cov_name)
    # Calculate the covariance matrix
    cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat + noise_cov_mat)
    # load_input_map
    x_map = vcat(set_params.Q_map + noise_map[1], set_params.U_map + noise_map[2])
    npix = hp.nside2npix(set_params.nside)
    xx_map = [x_map[1:npix]'.*0; x_map[1:npix]'; x_map[npix + 1:2 * npix]']
    smootinhg_x_map = smoothing_map_fwhm(xx_map, 2200, set_params.nside) 
    smoosthing_masked_x_map_Q = extract_masked_values(set_params, smootinhg_x_map[2,:] + art_noise_map[1])
    smoosthing_masked_x_map_U = extract_masked_values(set_params, smootinhg_x_map[3,:] + art_noise_map[2])
    x_sm_msked = [smoosthing_masked_x_map_Q ; smoosthing_masked_x_map_U]
    det_C = cholesky_logdet(cov_mat)
    return x_sm_msked' / cov_mat * x_sm_msked + det_C
end

function iterative_minimization_add_noise(set_params::SetParams, fit_params::FitParams, freq::Int)
    # initialize parameters
    r_pre = Inf
    r_out = 0.5
    neg2L_pre = -Inf
    neg2L_out = 1e10
    # loop until convergence
    while !(abs(neg2L_pre - neg2L_out) <= 1e-2 && abs(r_pre - r_out) <= 1e-5)
        # minimize likelihood_wrapper for r
        function likelihood_wrapper(r::Vararg{Float64})
            fit_params.r_est = r[1]
            return calc_likelihood_for_input_add_noise(set_params, fit_params, freq)
        end
        minuit_r = iminuit.Minuit(likelihood_wrapper, fit_params.r_est)
        minuit_r[:limits] = [(0.0, 1.0)]
        minuit_r[:errordef] = 1
        minuit_r.migrad()
        r_out = minuit_r[:values][1]
        # reset fg parameters
        r_pre = fit_params.r_est
        fit_params.r_est = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood_for_input_add_noise(set_params, fit_params, freq)
    end
    # return the fitted r value
    return fit_params.r_est
end