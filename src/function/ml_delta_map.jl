# using maximum likelihood Delta-map method to calculate likelihood function and Clean CMB map
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
using SparseArrays
using SymPy
@pyimport healpy as hp
@pyimport numpy as np

mutable struct SetParams
    freq_bands::Vector{Int}
    which_model::String
    r_input::Float64
    seed::Int
    nside::Int
    num_I::Int
    cov_mat_scal::Matrix{Float64}
    cov_mat_tens::Matrix{Float64}
    mask::Vector{Float64}
    m_set::Vector{Vector{Float64}}  
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

mutable struct InputMap
    cmb_Q::Vector{Float64}
    cmb_U::Vector{Float64}
    """Input Parameters"""
end

function calc_D_elements(fit_params::FitParams, freq)
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

function D_list_with_cmb_set(set_params::SetParams, fit_params::FitParams)
    """Set D list"""
    D_list = Vector{Matrix{Float64}}()
    for nu_i in set_params.freq_bands
        freq = nu_i * 10^9
        s, ss, d, dd, ddd = calc_D_elements(fit_params, freq)
        if set_params.which_model == "s1"
            D = [1, s, ss]
        elseif set_params.which_model == "d1" || set_params.which_model == "d10"
            D = [1, d, dd, ddd]
        elseif set_params.which_model == "d1 and s1"
            D = [1, s, ss, d, dd, ddd]
        end
        push!(D_list, hcat(D...))
    end
    return D_list
end

function smoothing_map_fwhm(input_map, input_fwhm, nside)
    """Smoothing map with input_fwhm"""
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    smoothed_map = hp.sphtfunc.alm2map(alm, nside, lmax=2*nside, pixwin=true, verbose=false, fwhm=input_fwhm*pi/10800.)
    return smoothed_map
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
        #println("Matrix A is not positive definite. Contents of A:")
        #println(A)
        #throw(ArgumentError("Matrix is not positive definite. Contents of A:\n$A"))
        throw(ArgumentError("Matrix is not positive definite."))
    end
    # Perform Cholesky decomposition and get the lower triangular matrix
    A_cho = cholesky(A).L
    return A_cho
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

function extract_masked_values(set_params::SetParams, map::Vector{Float64})
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

function set_N⁻¹_ml!(set_params::SetParams, noise)
    """
    Set N⁻¹ without noise 
    """
    npix = hp.nside2npix(set_params.nside)
    N⁻¹_set = []
    # artificial noise cov_mat
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    if noise == false
        for nu_i in set_params.freq_bands
            # set uniform cov_mat
            noise_cov_inv =  spdiagm(0 => ones(2npix)) 
            push!(N⁻¹_set, noise_cov_inv)   
        end
    else
        for nu_i in set_params.freq_bands
            # calc Noise cov_mat inverse
            freq_name = string(nu_i)
            dir_noise = "../map_file/noise_map/smoothing_noise_cov_mat/"
            nside_name = "nside_"
            nside_n = string(set_params.nside)
            # Load noise cov_mat
            noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, "_noise_cov_mat_with_smoothing.npy")
            noise_cov_mat = npzread(noise_cov_name)
            # masked noise cov_mat
            noise_cov_inv = positive_definite_inverse(noise_cov_mat + art_noise_cov_mat)
            push!(N⁻¹_set, noise_cov_inv)   
        end
    end
    set_params.N⁻¹_set = N⁻¹_set
end

function set_num_I_ml!(set_params::SetParams)
    if set_params.which_model == "s1"
        set_params.num_I = 3
    elseif set_params.which_model == "d1" || set_params.which_model == "d10"
        set_params.num_I = 4
    elseif set_params.which_model == "d1 and s1"
        set_params.num_I = 6
    else
        throw(ArgumentError("Invalid model type"))
    end
end

function calc_DᵀN⁻¹D_element_ml(set_params::SetParams, I, J, D_list)
    npix = hp.nside2npix(set_params.nside)
    DᵀN⁻¹D_element = zeros(2npix, 2npix)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        DᵀN⁻¹D_element .+= D_list[i][I] * set_params.N⁻¹_set[i] * D_list[i][J]
    end
    return DᵀN⁻¹D_element
end

function calc_DᵀN⁻¹D_ml(set_params::SetParams, D_list)
    npix = hp.nside2npix(set_params.nside)
    DᵀN⁻¹D = zeros(2npix * set_params.num_I, 2npix * set_params.num_I)
    @views for I in 1:set_params.num_I
        @inbounds for J in 1:set_params.num_I
            DᵀN⁻¹D[2npix*(I-1)+1:2npix*I, 2npix*(J-1)+1:2npix*J] = calc_DᵀN⁻¹D_element_ml(set_params, I, J, D_list)
        end
    end
    return DᵀN⁻¹D
end

function calc_DᵀN⁻¹m_element_ml(set_params::SetParams, I, D_list)
    npix = hp.nside2npix(set_params.nside)
    DᵀN⁻¹m_element = zeros(2npix)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Calculate DNm for each frequency element
        DᵀN⁻¹m_element .+= D_list[i][I] * set_params.N⁻¹_set[i] * set_params.m_set[i]       
    end
    return DᵀN⁻¹m_element
end

function calc_DᵀN⁻¹m_ml(set_params::SetParams, D_list)
    npix = hp.nside2npix(set_params.nside)
    DᵀN⁻¹m = zeros(2npix * set_params.num_I)
    for I in 1:set_params.num_I
        DᵀN⁻¹m[2npix*(I-1)+1:2npix*I] = calc_DᵀN⁻¹m_element_ml(set_params, I, D_list)
    end
    return DᵀN⁻¹m
end

function calc_clean_map_cmb_ml(set_params::SetParams, fit_params::FitParams)
    npix = hp.nside2npix(set_params.nside)
    D_list = D_list_with_cmb_set(set_params, fit_params)
    DᵀN⁻¹D = calc_DᵀN⁻¹D_ml(set_params, D_list)
    DᵀN⁻¹m = calc_DᵀN⁻¹m_ml(set_params, D_list)
    est_signal = DᵀN⁻¹D \ DᵀN⁻¹m
    clean_cmb_Q_sm = est_signal[1:npix]
    clean_cmb_U_sm = est_signal[npix+1:2npix]
    return clean_cmb_Q_sm, clean_cmb_U_sm
end

function calc_noise_res(set_params::SetParams, fit_params::FitParams)
    npix = hp.nside2npix(set_params.nside)
    D_list = D_list_with_cmb_set(set_params, fit_params)
    DᵀN⁻¹D = calc_DᵀN⁻¹D_ml(set_params, D_list)
    DᵀN⁻¹D = (DᵀN⁻¹D' + DᵀN⁻¹D) / 2
    N_res⁻¹ = positive_definite_inverse(DᵀN⁻¹D)
    N_res⁻¹_cmb = N_res⁻¹[1:2npix, 1:2npix]
    return N_res⁻¹_cmb
end

# pixel based likelihood
function calc_all_cov_mat(set_params::SetParams, fit_params::FitParams)
    """
    Calculate sum of covariance matrix.
    """
    cov_mat = set_params.cov_mat_scal + fit_params.r_est * set_params.cov_mat_tens 
    return cov_mat
end

function calc_pix_based_chi_sq(set_params::SetParams, fit_params::FitParams, noise::Bool)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    pol_sen = 0.2 # μK  
    # if it is not fixed to the seed, the result will be different each time
    Random.seed!(set_params.seed)
    # Random.seed!(set_params.seed) Because the seed is set in the main function
    art_noise_map, _ = calc_noise_map(pol_sen, set_params.nside)
    art_noise_cov_mat = calc_noise_cov_mat(pol_sen, set_params.nside)
    # Calculate the clean map
    if noise == false
        # Calculate the covariance matrix
        cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat)
        clean_Q, clean_U = calc_clean_map_cmb_ml(set_params, fit_params)
    else
        # calculate residual noise cov_mat
        N_res⁻¹ = calc_noise_res(set_params, fit_params)
        # Calculate the covariance matrix
        cov_mat = extract_masked_elements(set_params, calc_all_cov_mat(set_params, fit_params) + art_noise_cov_mat + N_res⁻¹)
        clean_Q, clean_U = calc_clean_map_cmb_ml(set_params, fit_params)
    end
    x_sm_masked = [extract_masked_values(set_params, clean_Q .+ art_noise_map.Q) ; extract_masked_values(set_params, clean_U .+ art_noise_map.U)]
    return x_sm_masked' / cov_mat * x_sm_masked, cov_mat 
end

function calc_pix_based_likelihood(set_params::SetParams, fit_params::FitParams, noise::Bool)
    # Calculate the covariance matrix
    chi_sq, cov_mat = calc_pix_based_chi_sq(set_params, fit_params, noise)
    det_C = cholesky_logdet(cov_mat)
    return chi_sq + det_C
end

#=========================== considering 2nd order purtubation ==================================#
function calc_D_elements_2nd(fit_params::FitParams, freq)
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
        ν => freq,
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

function D_list_2nd(set_params::SetParams, fit_params::FitParams)
    """Set D list"""
    D_list = Vector{Matrix{Float64}}()
    for nu_i in set_params.freq_bands
        freq = nu_i * 10^9
        s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd = calc_D_elements_2nd(fit_params, freq)
        if set_params.which_model == "s1"
            D = [1, s, ss, sss]
        elseif set_params.which_model == "d1" || set_params.which_model == "d10"
            D = [1, d, dd, ddd, dddd, ddddd, dddddd]
        elseif set_params.which_model == "d1 and s1"
            D = [1, s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd]
        end
        push!(D_list, hcat(D...))
    end
    return D_list
end

function set_num_I_2nd!(set_params::SetParams)
    if set_params.which_model == "s1"
        set_params.num_I = 4
    elseif set_params.which_model == "d1" || set_params.which_model == "d10"
        set_params.num_I = 7
    elseif set_params.which_model == "d1 and s1"
        set_params.num_I = 10
    else
        throw(ArgumentError("Invalid model type"))
    end
end

function calc_clean_map_cmb_2nd(set_params::SetParams, fit_params::FitParams)#, noise)
    # npix depends on the mask
    #npix = count(x -> x == 1, set_params.mask)
    npix = hp.nside2npix(set_params.nside)
    D_list = D_list_2nd(set_params, fit_params)
    DᵀN⁻¹D = calc_DᵀN⁻¹D_ml(set_params, D_list)
    DᵀN⁻¹m = calc_DᵀN⁻¹m_ml(set_params, D_list)
    est_signal = DᵀN⁻¹D \ DᵀN⁻¹m
    clean_cmb_Q_sm = est_signal[1:npix]
    clean_cmb_U_sm = est_signal[npix+1:2npix]
    return clean_cmb_Q_sm, clean_cmb_U_sm
end