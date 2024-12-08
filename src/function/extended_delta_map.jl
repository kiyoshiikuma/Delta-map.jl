# # using Extended Delta-map method to calculate likelihood function and Clean CMB map
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
using SparseArrays
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

function D_list_set(set_params::SetParams, fit_params::FitParams)
    """Set D list"""
    D_list = Vector{Matrix{Float64}}()
    for nu_i in set_params.freq_bands
        freq = nu_i * 10^9
        s, ss, d, dd, ddd = calc_D_elements(fit_params, freq)
        if set_params.which_model == "s1"
            D = [s, ss]
        elseif set_params.which_model == "d1"
            D = [d, dd, ddd]
        elseif set_params.which_model == "d1 and s1"
            D = [s, ss, d, dd, ddd]
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

function set_N⁻¹!(set_params::SetParams)
    """
    Set N⁻¹
    """
    N⁻¹_set = []
    # artificial noise cov_mat
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
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
        noise_cov_inv = positive_definite_inverse(extract_masked_elements(set_params, noise_cov_mat + art_noise_cov_mat))
        push!(N⁻¹_set, noise_cov_inv)   
    end
    set_params.N⁻¹_set = N⁻¹_set
end

function set_num_I!(set_params::SetParams)
    if set_params.which_model == "s1"
        set_params.num_I = 2
    elseif set_params.which_model == "d1"
        set_params.num_I = 3
    elseif set_params.which_model == "d1 and s1"
        set_params.num_I = 5
    else
        throw(ArgumentError("Invalid model type"))
    end
end

function calc_DᵀN⁻¹D_element(set_params::SetParams, I, J, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹D_element = zeros(2npix, 2npix)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        DᵀN⁻¹D_element .+= D_list[i][I] * set_params.N⁻¹_set[i] * D_list[i][J]
    end
    return DᵀN⁻¹D_element
end

function calc_DᵀN⁻¹D(set_params::SetParams, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹D = zeros(2npix * set_params.num_I, 2npix * set_params.num_I)
    @views for I in 1:set_params.num_I
        @inbounds for J in 1:set_params.num_I
            DᵀN⁻¹D[2npix*(I-1)+1:2npix*I, 2npix*(J-1)+1:2npix*J] = calc_DᵀN⁻¹D_element(set_params, I, J, D_list)
        end
    end

    return DᵀN⁻¹D
end

function calc_DᵀN⁻¹m_element(set_params::SetParams, I, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹m_element = zeros(2npix)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Calculate DNm for each frequency element
        DᵀN⁻¹m_element .+= D_list[i][I] * set_params.N⁻¹_set[i] * set_params.m_set[i]       
    end
    return DᵀN⁻¹m_element
end

function calc_DᵀN⁻¹m(set_params::SetParams, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹m = zeros(2npix * set_params.num_I)
    for I in 1:set_params.num_I
        DᵀN⁻¹m[2npix*(I-1)+1:2npix*I] = calc_DᵀN⁻¹m_element(set_params, I, D_list)
    end
    return DᵀN⁻¹m
end

function calc_A(set_params::SetParams, fit_params::FitParams)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    ΣN⁻¹ = zeros(2npix, 2npix)
    # Calculate ΣN⁻¹
    for i in 1:length(set_params.freq_bands)
        ΣN⁻¹ .+= set_params.N⁻¹_set[i]
    end
    art_noise = calc_noise_cov_mat(0.2, set_params.nside)
    cov_cmb = set_params.cov_mat_scal + fit_params.r_est[1] * set_params.cov_mat_tens
    cov_cmb⁻¹ = positive_definite_inverse(extract_masked_elements(set_params, cov_cmb + art_noise))
    A = cov_cmb⁻¹ + ΣN⁻¹
    return A
end

function set_M_vec(set_params::SetParams, fit_params::FitParams)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    N⁻¹m_element = zeros(2npix)
    M_set = []
    # Calculate ΣN^-1m
    for i in 1:length(set_params.freq_bands)
        # Calculate DNm for each frequency element
        N⁻¹m_element .+= set_params.N⁻¹_set[i] * set_params.m_set[i]     
    end
    A = calc_A(set_params, fit_params)
    A_L = cholesky(A)
    M = A_L \ N⁻¹m_element
    for j in 1:length(set_params.freq_bands)
        # M_set for each frequency
        push!(M_set, set_params.m_set[j] - M)
    end
    return M_set
end

function calc_DᵀN⁻¹M_element(set_params::SetParams, fit_params::FitParams, I, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹M_element = zeros(2npix)
    M_vec_nu = set_M_vec(set_params, fit_params)
    # Calculate DᵀN⁻¹M
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Calculate DNm for each frequency element
        DᵀN⁻¹M_element .+= D_list[i][I] * set_params.N⁻¹_set[i] * M_vec_nu[i]       
    end
    return DᵀN⁻¹M_element
end

function calc_DᵀN⁻¹M(set_params::SetParams, fit_params::FitParams, D_list)
    # npix depends on the mask
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹M = zeros(2npix * set_params.num_I)
    for I in 1:set_params.num_I
        DᵀN⁻¹M[2npix*(I-1)+1:2npix*I] = calc_DᵀN⁻¹M_element(set_params, fit_params, I, D_list)
    end
    return DᵀN⁻¹M
end

function calc_DᵀN⁻¹Dcmb_element(set_params::SetParams, I::Int, D_list::Vector{Matrix{Float64}})
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹Dcmb_element = zeros(2npix, 2npix)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # calc DNm 各周波数要素の計算
        DᵀN⁻¹Dcmb_element .+= D_list[i][I] * set_params.N⁻¹_set[i]
    end
    return DᵀN⁻¹Dcmb_element
end

function calc_DᵀN⁻¹Dcmb(set_params::SetParams, D_list::Vector{Matrix{Float64}})
    npix = count(x -> x == 1, set_params.mask)
    DᵀN⁻¹Dcmb = zeros(2npix*set_params.num_I, 2npix)
    for I in 1:set_params.num_I
        DᵀN⁻¹Dcmb[2npix*(I-1)+1:2npix*I, 1:2npix] = calc_DᵀN⁻¹Dcmb_element(set_params, I, D_list)
    end
    return DᵀN⁻¹Dcmb
end

function calc_DcmbᵀN⁻¹m(set_params::SetParams)
    npix = count(x -> x == 1, set_params.mask)
    DcmbᵀN⁻¹m = zeros(2npix)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        DcmbᵀN⁻¹m .+= set_params.N⁻¹_set[i] * set_params.m_set[i]
    end
    return DcmbᵀN⁻¹m
end

function calc_B(set_params::SetParams, fit_params::FitParams, D_list)
    DᵀN⁻¹D = calc_DᵀN⁻¹D(set_params, D_list)
    DᵀN⁻¹Dcmb = calc_DᵀN⁻¹Dcmb(set_params, D_list)
    DcmbᵀHDcmb = DᵀN⁻¹Dcmb' * (DᵀN⁻¹D \ DᵀN⁻¹Dcmb)
    A = calc_A(set_params, fit_params)
    B = - DcmbᵀHDcmb + A
    return B
end

function calc_mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m(set_params::SetParams, fit_params::FitParams)
    DcmbᵀN⁻¹m = calc_DcmbᵀN⁻¹m(set_params)
    A = calc_A(set_params, fit_params)
    A_L = cholesky(A)
    mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m = DcmbᵀN⁻¹m' * (A_L \ DcmbᵀN⁻¹m)
    return mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m 
end

function calc_MᵀHM(set_params::SetParams, fit_params::FitParams, D_list)
    DᵀN⁻¹D = calc_DᵀN⁻¹D(set_params, D_list)
    DᵀN⁻¹M = calc_DᵀN⁻¹M(set_params, fit_params, D_list)
    MHM = DᵀN⁻¹M' * (DᵀN⁻¹D \ DᵀN⁻¹M)
    return MHM
end

function calc_MᵀHB⁻¹HM(set_params::SetParams, fit_params::FitParams, D_list::Vector{Matrix{Float64}})
    DᵀN⁻¹D = calc_DᵀN⁻¹D(set_params, D_list)
    DᵀN⁻¹Dcmb = calc_DᵀN⁻¹Dcmb(set_params, D_list)
    DᵀN⁻¹M = calc_DᵀN⁻¹M(set_params, fit_params, D_list)
    DcmbHM = DᵀN⁻¹Dcmb' * (DᵀN⁻¹D \ DᵀN⁻¹M)
    B = calc_B(set_params, fit_params, D_list)
    MᵀHB⁻¹HM = DcmbHM' * (B \ DcmbHM)
    return MᵀHB⁻¹HM
end

function calc_chi_sq(set_params::SetParams, fit_params::FitParams)
    D_list = D_list_set(set_params, fit_params)
    mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m = - calc_mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m(set_params, fit_params)
    MᵀHM = - calc_MᵀHM(set_params, fit_params, D_list)
    MᵀHB⁻¹HM = - calc_MᵀHB⁻¹HM(set_params, fit_params, D_list)
    chi_sq = mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m + MᵀHM + MᵀHB⁻¹HM
    return chi_sq
end

function calc_likelihood(set_params::SetParams, fit_params::FitParams)
    D_list = D_list_set(set_params, fit_params)
    chi_sq = calc_chi_sq(set_params, fit_params)
    art_noise = calc_noise_cov_mat(0.2, set_params.nside)
    cov_cmb = set_params.cov_mat_scal + fit_params.r_est[1] .* set_params.cov_mat_tens
    cov_cmb_art_noise = extract_masked_elements(set_params, cov_cmb + art_noise)
    b = calc_B(set_params, fit_params, D_list)
    lnScmb = cholesky_logdet(cov_cmb_art_noise)
    lnDᵀN⁻¹D = cholesky_logdet(calc_DᵀN⁻¹D(set_params, D_list))
    lnb = cholesky_logdet(b)
    log_det_part = lnScmb + lnDᵀN⁻¹D + lnb
    return chi_sq + log_det_part #, chi_sq, lnScmb, lnDᵀN⁻¹D, lnb
end