# using Extended Delta-map method to calculate likelihood function
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
    TᵀN⁻¹_set::Vector{Matrix{ComplexF64}}
    TᵀN⁻¹T_set::Vector{Matrix{ComplexF64}}
    spin::Int
    lmax_alm::Int
    """Parameters for r estimation"""
end

mutable struct FitParams
    beta_s::Float64
    beta_d::Float64
    T_d::Float64
    r_est::Float64
    """Optimization Parameters"""
end

mutable struct CholeskyTerms
    DᵀN⁻¹D_L::Cholesky{ComplexF64, Matrix{ComplexF64}}
    A_L::Cholesky{Float64, Matrix{Float64}}
    B_L::Cholesky{ComplexF64, Matrix{ComplexF64}}
    """Cholesky Decomposition"""
end

mutable struct MatrixTerms
    DᵀN⁻¹m::Vector{ComplexF64}
    DᵀN⁻¹Dcmb::Matrix{ComplexF64}
    A::Matrix{Float64}
    """Matrix Terms"""
end

mutable struct Sphenical_Harmonics
    T0::Matrix{ComplexF64}
    save_Wlm::Array{ComplexF64,3}
    save_Xlm::Array{ComplexF64,3}
    """
    Save Wlm, Xlm for each frequency band.
    The shape is (Npix, Nlm, Nfreq).
    """
end

function set_cholesky_terms!()
    """
    Initialize CholeskyTerm with dummy positive definite matrices.
    """
    dummy_matrix = Matrix{Float64}(I, 1, 1) 
    dummy_matrix_comp = Matrix{ComplexF64}(I, 1, 1) 
    DᵀN⁻¹D_L = cholesky(dummy_matrix_comp)
    A_L = cholesky(dummy_matrix)
    B_L = cholesky(dummy_matrix_comp)
    return CholeskyTerms(DᵀN⁻¹D_L, A_L, B_L)
end

function set_matrix_terms!()
    """
    Initialize MatrixTerms with empty matrices.
    """
    empty_vector = ComplexF64[] 
    empty_matrix = Array{ComplexF64, 2}(undef, 0, 0)  
    return MatrixTerms(
        empty_vector,  # DᵀN⁻¹m
        empty_matrix,  # DᵀN⁻¹Dcmb
        empty_matrix,   # A
    )
end

function set_sphenical_harmonics_terms!()
    """
    Initialize MatrixTerms with empty matrices.
    """
    empty_matrix_comp = Array{ComplexF64, 2}(undef, 0, 0)  
    empty_array_comp  = Array{ComplexF64, 3}(undef, 0, 0, 0)
    return Sphenical_Harmonics(
        empty_matrix_comp,  # T0
        empty_array_comp,   # Wlm
        empty_array_comp,   # Xlm_
    )
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

function set_num_I!(set_params::SetParams)
    if set_params.which_model == "s1"
        set_params.num_I = 2
    elseif set_params.which_model == "d1" || set_params.which_model == "d10"
        set_params.num_I = 3
    elseif set_params.which_model == "d1 and s1"
        set_params.num_I = 5
    else
        throw(ArgumentError("Invalid model type"))
    end
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
    # Symmetrize the matrix
    A = (A + A') / 2
    # Check if the matrix is positive definite
    # Perform Cholesky decomposition and get the lower triangular matrix
    A_cho = cholesky(A)
    return A_cho
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

function cholesky_logdet(A::Cholesky{T, S}) where {T, S <: AbstractMatrix}
    """
    Compute the log-determinant of a positive definite matrix using Cholesky decomposition.
    """
    # Perform Cholesky decomposition
    # Twice the log-determinant of the matrix
    logdet = 2 * sum(log.(diag(A.L)))
    return logdet
end

function extract_masked_values(set_params::SetParams, map::Vector{Float64})
    return [map[i] for i in 1:length(set_params.mask) if set_params.mask[i] == 1]
end

function set_truncate_N⁻¹!(set_params::SetParams, lmin, lmax)
    """
    Set N⁻¹
    """
    N⁻¹_set = []
    # artificial noise cov_mat
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)
    for nu_i in set_params.freq_bands
        # calc Noise cov_mat inverse
        freq_name = string(nu_i)
        dir_noise = "/Users/ikumakiyoshi/Library/Mobile Documents/com~apple~CloudDocs/study_fg_rm/program/julia_Delta_map/Delta_map/make_noise_covariance_matrix/smoothing_noise_cov_mat/"
        nside_name = "nside_"
        nside_n = string(set_params.nside)
        # Load noise cov_mat
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        #noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing.npy")
        #noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_40GHz_beam.npy")
        #noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_40GHz_beam_wo_wl.npy")
        #noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_pixsize25_beam.npy")
        #noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_pixsize25_beam_wo_wl.npy")
        noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_wo_beam_wo_wl.npy")
        noise_cov_mat = npzread(noise_cov_name)
        # masked noise cov_mat
        noise_cov_inv = positive_definite_inverse(extract_masked_elements(set_params, noise_cov_mat + art_noise_cov_mat))
        #II = noise_cov_inv * extract_masked_elements(set_params, noise_cov_mat + art_noise_cov_mat)
        #println("Inverse check for freq $freq_name: ", isapprox(II, LinearAlgebra.I, rtol=1e-12, atol=1e-14))
        #@assert isapprox(II, LinearAlgebra.I, rtol=1e-6, atol=1e-8) "Inverse check failed for freq $freq_name"
        push!(N⁻¹_set, noise_cov_inv)   
    end
    set_params.N⁻¹_set = N⁻¹_set
end

function set_T0_matrix(set_params::SetParams, sphenical_harmonics::Sphenical_Harmonics, Wmat::Matrix{ComplexF64}, Xmat::Matrix{ComplexF64}, mask_name)
    """
    Set T0 matrix for the given set parameters and saved Wlm, Xlm.
    """
    # T0 = build_T0_masked(set_params, sphenical_harmonics)
    T0 = build_T0_masked(set_params, Wmat, Xmat)
    sphenical_harmonics.T0 = T0
    namewo = basename(mask_path)
    npzwrite("T_matrix/T0_$(namewo)_nside_$(set_params.nside)_lmin_$(2)_lmax_$(set_params.lmax_alm)", T0)
    return T0
end

#================ calculate WXlm ====================#
function lm_list(lmax::Int, spin::Int)
    ℓ_min = max(abs(spin), 2)
    [(ℓ,m) for ℓ in ℓ_min:lmax for m in -ℓ:ℓ]
end

function get_vec_WXlm(save_WXlm::Array{ComplexF64,3}, ℓ::Int, m::Int)
    _, Ldim, _ = size(save_WXlm)
    ℓ_max = Ldim - 1
    mi = m + ℓ_max + 1
    li = ℓ + 1
    return @view save_WXlm[:, li, mi]
end

function make_WXmat_from_save(set_params::SetParams, save_WXlm::Array{ComplexF64,3})
    npix, Ldim, _ = size(save_WXlm)
    #lmax = Ldim - 1
    pairs = lm_list(set_params.lmax_alm, set_params.spin)
    WXmat = zeros(ComplexF64, npix, length(pairs))
    for (j, (ℓ,m)) in enumerate(pairs)
        #println("ℓ: ", ℓ, " m: ", m)
        WXmat[:, j] = get_vec_WXlm(save_WXlm, ℓ, m)
    end
    return WXmat
end

# function build_T0_masked(set_params::SetParams, sphenical_harmonics::Sphenical_Harmonics)
function build_T0_masked(set_params::SetParams, Wmat::Matrix{ComplexF64}, Xmat::Matrix{ComplexF64})
    #Wmat = make_WXmat_from_save(set_params, sphenical_harmonics.save_Wlm)      # Npix × Nlm
    #Xmat = make_WXmat_from_save(set_params, sphenical_harmonics.save_Xlm)      # Npix × Nlm
    # println("Wmat size: ", size(Wmat))
    # println("Xmat size: ", size(Xmat))
    @assert length(set_params.mask) == size(Wmat,1)
    mask_bool = set_params.mask .!= 0                     # 1→true, 0→false
    Wm = Wmat[mask_bool, :]                      # Npix_masked × Nlm
    Xm = Xmat[mask_bool, :]                      # Npix_masked × Nlm

    return [ Wm  Xm;
            -Xm   Wm ]                           # (2*Npixm)×(2*Nlm)
end

# save TᵀN⁻¹T, TᵀN⁻¹
function calc_TᵀN⁻¹T_terms(set_params::SetParams, mask_name, nside, lmin, lmax)
    # npix depends on the mask
    for (i, nu_i) in enumerate(set_params.freq_bands)
        TᵀN⁻¹ = sphenical_harmonics.T0' * set_params.N⁻¹_set[i]
        TᵀN⁻¹T = sphenical_harmonics.T0' * set_params.N⁻¹_set[i] * sphenical_harmonics.T0
        namewo = basename(mask_path)
        #npzwrite("T_N_inv/TᵀN⁻¹_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_40GHz_beam_wo_wl.npy", TᵀN⁻¹)
        #npzwrite("T_N_inv_T/TᵀN⁻¹T_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_40GHz_beam_wo_wl.npy", TᵀN⁻¹T)
        #npzwrite("T_N_inv/TᵀN⁻¹_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_pixsize25_beam_wo_wl.npy", TᵀN⁻¹)
        #npzwrite("T_N_inv_T/TᵀN⁻¹T_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_pixsize25_beam_wo_wl.npy", TᵀN⁻¹T)
        npzwrite("T_N_inv/TᵀN⁻¹_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_wo_beam_wo_wl.npy", TᵀN⁻¹)
        npzwrite("T_N_inv_T/TᵀN⁻¹T_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_wo_beam_wo_wl.npy", TᵀN⁻¹T)
    end
end