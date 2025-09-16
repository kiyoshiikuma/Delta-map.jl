# using Extended Delta-map method to calculate likelihood function
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
using SymPy
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
    T0_set::Matrix{ComplexF64}
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
    A_L::Cholesky{ComplexF64, Matrix{ComplexF64}}
    B_L::Cholesky{ComplexF64, Matrix{ComplexF64}}
    """Cholesky Decomposition"""
end

mutable struct MatrixTerms
    DᵀN⁻¹m::Vector{ComplexF64}
    DᵀN⁻¹Dcmb::Matrix{ComplexF64}
    A::Matrix{ComplexF64}
    """Matrix Terms"""
end

mutable struct Sphenical_Harmonics
    T0::Matrix{ComplexF64}
    #save_Wlm::Array{ComplexF64,3}
    #save_Xlm::Array{ComplexF64,3}
    """
    Save Wlm, Xlm for each frequency band.
    The shape is (Npix, Nlm, Nfreq).
    """
end

function set_cholesky_terms!()
    """
    Initialize CholeskyTerm with dummy positive definite matrices.
    """
    dummy_matrix_comp = Matrix{ComplexF64}(I, 1, 1) 
    DᵀN⁻¹D_L = cholesky(dummy_matrix_comp)
    A_L = cholesky(dummy_matrix_comp)
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
        #empty_array_comp,   # Wlm
        #empty_array_comp,   # Xlm_
    )
end

function set_T0_matrix(set_params::SetParams, sphenical_harmonics::Sphenical_Harmonics, Wmat::Matrix{ComplexF64}, Xmat::Matrix{ComplexF64})
    """
    Set T0 matrix for the given set parameters and saved Wlm, Xlm.
    """
    # T0 = build_T0_masked(set_params, sphenical_harmonics)
    T0 = build_T0_masked(set_params, Wmat, Xmat)
    sphenical_harmonics.T0 = T0
    return T0
end

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

function D_list_set(set_params::SetParams, fit_params::FitParams)
    """Set D list"""
    D_list = Vector{Matrix{Float64}}()
    for nu_i in set_params.freq_bands
        freq = nu_i * 10^9
        s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd = calc_D_elements(fit_params, freq)
        if set_params.which_model == "s1" || set_params.which_model == "s5" 
            #D = [s, ss, sss]
            D = [s, ss]
        elseif set_params.which_model == "d1" || set_params.which_model == "d10"
            #D = [d, dd, ddd, dddd, ddddd, dddddd]
            D = [d, dd, ddd]
        elseif set_params.which_model == "d1 and s1"
            #D = [s, ss, sss, d, dd, ddd, dddd, ddddd, dddddd]
            D = [s, ss, d, dd, ddd]
        end
        push!(D_list, hcat(D...))
    end
    return D_list
end

function set_num_I!(set_params::SetParams)
    if set_params.which_model == "s1" || set_params.which_model == "s5"
        #set_params.num_I = 3
        set_params.num_I = 2
    elseif set_params.which_model == "d1" || set_params.which_model == "d10"
        #set_params.num_I = 6
        set_params.num_I = 3
    elseif set_params.which_model == "d1 and s1" || set_params.which_model == "d10 and s5"
        #set_params.num_I = 9
        set_params.num_I = 5
    else
        throw(ArgumentError("Invalid model type"))
    end
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

function set_truncate_TᵀN⁻¹_TᵀN⁻¹T!(set_params::SetParams, mask_path)
    """
    Set TᵀN⁻¹, TᵀN⁻¹T
    """
    TᵀN⁻¹_set = []
    TᵀN⁻¹T_set = []
    for nu_i in set_params.freq_bands
        # Load TᵀN⁻¹, TᵀN⁻¹T
        namewo = basename(mask_path)
        TᵀN⁻¹ = npzread("../../T_N_inv/TᵀN⁻¹_$(namewo)_nside_$(set_params.nside)_freq_$(nu_i)_lmin_$(2)_lmax_$(set_params.lmax_alm)_40GHz_beam_wo_wl.npy")
        TᵀN⁻¹T = npzread("../../T_N_inv_T/TᵀN⁻¹T_$(namewo)_nside_$(set_params.nside)_freq_$(nu_i)_lmin_$(2)_lmax_$(set_params.lmax_alm)_40GHz_beam_wo_wl.npy")
        push!(TᵀN⁻¹_set, TᵀN⁻¹)
        push!(TᵀN⁻¹T_set, TᵀN⁻¹T)
    end
    namewo = basename(mask_path)
    T0_set = npzread("../../T_matrix/T0_$(namewo)_nside_$(set_params.nside)_lmin_$(2)_lmax_$(set_params.lmax_alm)")
    set_params.TᵀN⁻¹_set = TᵀN⁻¹_set
    set_params.TᵀN⁻¹T_set = TᵀN⁻¹T_set
    set_params.T0_set = T0_set
end

function calc_DᵀN⁻¹D_element(set_params::SetParams, I, J, D_list)
    """
    Calculate DᵀN⁻¹D element
    """
    # Nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics 
    DᵀN⁻¹D_element = zeros(ComplexF64, Nlm, Nlm)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        DᵀN⁻¹D_element .+= D_list[i][I]' * set_params.TᵀN⁻¹T_set[i] * D_list[i][J]
    end
    return DᵀN⁻¹D_element
end

function calc_DᵀN⁻¹D(set_params::SetParams, cholesky_terms::CholeskyTerms, D_list)
    # nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics 
    DᵀN⁻¹D = zeros(ComplexF64, Nlm * set_params.num_I, Nlm * set_params.num_I)
    @views for I in 1:set_params.num_I
        @inbounds for J in 1:set_params.num_I
            DᵀN⁻¹D[Nlm*(I-1)+1:Nlm*I, Nlm*(J-1)+1:Nlm*J] = calc_DᵀN⁻¹D_element(set_params, I, J, D_list)
        end
    end
    #DᵀN⁻¹D = real.(DᵀN⁻¹D)
    #npzwrite("/Users/ikumakiyoshi/Desktop/DᵀN⁻¹D.npy", DᵀN⁻¹D)
    cholesky_terms.DᵀN⁻¹D_L = cholesky_decomposition(DᵀN⁻¹D)
    return DᵀN⁻¹D
end

function calc_DᵀN⁻¹m_element(set_params::SetParams, I, D_list)
    # nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1) # 2 times the number of spherical harmonics 
    DᵀN⁻¹m_element = zeros(ComplexF64, Nlm)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Calculate DNm for each frequency element
        DᵀN⁻¹m_element .+= D_list[i][I]' * set_params.TᵀN⁻¹_set[i] * set_params.m_set[i]       
    end
    return DᵀN⁻¹m_element
end

function calc_DᵀN⁻¹m(set_params::SetParams, matrix_terms::MatrixTerms, D_list)
    # nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics 
    DᵀN⁻¹m = zeros(ComplexF64, Nlm * set_params.num_I)
    for I in 1:set_params.num_I
        DᵀN⁻¹m[Nlm*(I-1)+1:Nlm*I] = calc_DᵀN⁻¹m_element(set_params, I, D_list)
    end
    matrix_terms.DᵀN⁻¹m = DᵀN⁻¹m
    return DᵀN⁻¹m
end

function calc_A(set_params::SetParams, fit_params::FitParams, cholesky_terms::CholeskyTerms, matrix_terms::MatrixTerms)
    # Nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)
    ΣTᵀN⁻¹T = zeros(ComplexF64, Nlm, Nlm)
    # Calculate ΣTᵀN⁻¹T
    for i in 1:length(set_params.freq_bands)
        ΣTᵀN⁻¹T .+= set_params.TᵀN⁻¹T_set[i]
    end
    art_noise = calc_noise_cov_mat(0.2, set_params.nside)
    cov_cmb = set_params.cov_mat_scal + fit_params.r_est[1] * set_params.cov_mat_tens
    cov_cmb⁻¹ = positive_definite_inverse(extract_masked_elements(set_params, cov_cmb + art_noise))
    A = set_params.T0_set' * cov_cmb⁻¹ * set_params.T0_set + ΣTᵀN⁻¹T
    matrix_terms.A = A
    cholesky_terms.A_L = cholesky_decomposition(A)
end

function set_M_vec(set_params::SetParams, cholesky_terms::CholeskyTerms)
    # Nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)
    TᵀN⁻¹m_element = zeros(ComplexF64, Nlm)
    M_set = []
    # Calculate ΣN^-1m
    for i in 1:length(set_params.freq_bands)
        # Calculate DNm for each frequency element
        TᵀN⁻¹m_element .+= set_params.TᵀN⁻¹_set[i] * set_params.m_set[i]     
    end
    # M = set_params.T0_set * (cholesky_terms.A_L \ set_params.T0_set') * N⁻¹m_element
    M = set_params.T0_set * (cholesky_terms.A_L \ TᵀN⁻¹m_element)
    for j in 1:length(set_params.freq_bands)
        # M_set for each frequency
        push!(M_set, set_params.m_set[j] - M)
    end
    return M_set
end

function calc_DᵀN⁻¹M_element(set_params::SetParams, cholesky_terms::CholeskyTerms, I, D_list)
    # nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics
    DᵀN⁻¹M_element = zeros(ComplexF64, Nlm)
    M_vec_nu = set_M_vec(set_params, cholesky_terms)
    # Calculate DᵀN⁻¹M
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Calculate DNm for each frequency element
        DᵀN⁻¹M_element .+= D_list[i][I]' * set_params.TᵀN⁻¹_set[i] * M_vec_nu[i]       
    end
    return DᵀN⁻¹M_element
end

function calc_DᵀN⁻¹M(set_params::SetParams, cholesky_terms::CholeskyTerms, D_list)
    # nlm depends on the lmax
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics
    DᵀN⁻¹M = zeros(ComplexF64, Nlm * set_params.num_I)
    for I in 1:set_params.num_I
        DᵀN⁻¹M[Nlm*(I-1)+1:Nlm*I] = calc_DᵀN⁻¹M_element(set_params, cholesky_terms, I, D_list)
    end
    return DᵀN⁻¹M
end

function calc_DᵀN⁻¹Dcmb_element(set_params::SetParams, I::Int, D_list::Vector{Matrix{Float64}})
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)  # 2 times the number of spherical harmonics 
    DᵀN⁻¹Dcmb_element = zeros(ComplexF64, Nlm, Nlm)
    # D^T*N^-1*D
    # println(D_list)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # calc DNm 各周波数要素の計算
        DᵀN⁻¹Dcmb_element .+= D_list[i][I]' * set_params.TᵀN⁻¹T_set[i]
    end
    return DᵀN⁻¹Dcmb_element
end

function calc_DᵀN⁻¹Dcmb(set_params::SetParams, matrix_terms::MatrixTerms, D_list::Vector{Matrix{Float64}})
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1) # 2 times the number of spherical harmonics 
    DᵀN⁻¹Dcmb = zeros(ComplexF64, Nlm*set_params.num_I, Nlm)
    for I in 1:set_params.num_I
        DᵀN⁻¹Dcmb[Nlm*(I-1)+1:Nlm*I, 1:Nlm] = calc_DᵀN⁻¹Dcmb_element(set_params, I, D_list)
    end
    matrix_terms.DᵀN⁻¹Dcmb = DᵀN⁻¹Dcmb
    return DᵀN⁻¹Dcmb
end

function calc_DcmbᵀN⁻¹m(set_params::SetParams)
    Nlm = size(set_params.TᵀN⁻¹T_set[1],1)
    DcmbᵀN⁻¹m = zeros(ComplexF64, Nlm)
    # D^T*N^-1*D
    for (i, nu_i) in enumerate(set_params.freq_bands)
        DcmbᵀN⁻¹m .+= set_params.TᵀN⁻¹_set[i] * set_params.m_set[i]
    end
    return DcmbᵀN⁻¹m
end

function calc_B(cholesky_terms::CholeskyTerms)
    DcmbᵀHDcmb = matrix_terms.DᵀN⁻¹Dcmb' * (cholesky_terms.DᵀN⁻¹D_L \ matrix_terms.DᵀN⁻¹Dcmb)
    B = - DcmbᵀHDcmb + matrix_terms.A
    cholesky_terms.B_L = cholesky_decomposition(B)
end

function calc_mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m(set_params::SetParams)
    DcmbᵀN⁻¹m = calc_DcmbᵀN⁻¹m(set_params)
    mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m = DcmbᵀN⁻¹m' * (cholesky_terms.A_L \ DcmbᵀN⁻¹m)
    return mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m 
end

function calc_MᵀHM_and_MᵀHB⁻¹HM(set_params::SetParams, cholesky_terms::CholeskyTerms, D_list::Vector{Matrix{Float64}})
    DᵀN⁻¹M = calc_DᵀN⁻¹M(set_params, cholesky_terms, D_list)
    inv_DᵀN⁻¹D_DᵀN⁻¹M = cholesky_terms.DᵀN⁻¹D_L \ DᵀN⁻¹M
    MHM = DᵀN⁻¹M' * inv_DᵀN⁻¹D_DᵀN⁻¹M
    DcmbHM = matrix_terms.DᵀN⁻¹Dcmb' * inv_DᵀN⁻¹D_DᵀN⁻¹M
    MᵀHB⁻¹HM = DcmbHM' * (cholesky_terms.B_L \ DcmbHM)
    return MHM, MᵀHB⁻¹HM
end

function calc_all_terms(set_params::SetParams, fit_params::FitParams, cholesky_terms::CholeskyTerms, matrix_terms::MatrixTerms, D_list::Vector{Matrix{Float64}})
    calc_DᵀN⁻¹Dcmb(set_params, matrix_terms, D_list)
    calc_DᵀN⁻¹m(set_params, matrix_terms, D_list)
    calc_A(set_params, fit_params, cholesky_terms, matrix_terms)
    calc_DᵀN⁻¹D(set_params, cholesky_terms, D_list)
    calc_B(cholesky_terms)
end

function calc_chi_sq(set_params::SetParams, fit_params::FitParams, cholesky_terms::CholeskyTerms)
    D_list = D_list_set(set_params, fit_params)
    calc_all_terms(set_params, fit_params, cholesky_terms, matrix_terms, D_list)
    mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m = calc_mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m(set_params)
    MᵀHM, MᵀHB⁻¹HM = calc_MᵀHM_and_MᵀHB⁻¹HM(set_params, cholesky_terms, D_list)
    chi_sq = - mN⁻¹DcmbA⁻¹DcmbᵀN⁻¹m - MᵀHM - MᵀHB⁻¹HM
    return real.(chi_sq)
end

function calc_likelihood(set_params::SetParams, fit_params::FitParams, cholesky_terms::CholeskyTerms)
    """
    Calculate the likelihood function for the given set parameters and fit parameters.
    """
    chi_sq = calc_chi_sq(set_params, fit_params, cholesky_terms)
    art_noise = calc_noise_cov_mat(0.2, set_params.nside)
    cov_cmb = set_params.cov_mat_scal + fit_params.r_est[1] .* set_params.cov_mat_tens
    cov_cmb⁻¹ = positive_definite_inverse(extract_masked_elements(set_params, cov_cmb + art_noise))
    #cov_cmb_art_noise = extract_masked_elements(set_params, art_noise) + cov_cmb
    lnTᵀScmbT = cholesky_logdet(cholesky_decomposition(set_params.T0_set' * cov_cmb⁻¹ * set_params.T0_set))
    lnDᵀN⁻¹D = cholesky_logdet(cholesky_terms.DᵀN⁻¹D_L)
    lnb = cholesky_logdet(cholesky_terms.B_L)
    log_det_part = - lnTᵀScmbT + lnDᵀN⁻¹D + lnb
    return real.(chi_sq + log_det_part)
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

function build_T(sphenical_harmonics::Sphenical_Harmonics, f::Float64)
    return f .* sphenical_harmonics.T0    
end

# save TᵀN⁻¹T, TᵀN⁻¹
function calc_TᵀN⁻¹T_terms(set_params::SetParams, mask_name, nside, lmin, lmax)
    # npix depends on the mask
    Nlm = size(D_list[1][1], 2)  # 2 times the number of spherical harmonics 
    DᵀN⁻¹D_element = zeros(ComplexF64, Nlm, Nlm)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        TᵀN⁻¹ = sphenical_harmonics.T0' * set_params.N⁻¹_set[i]
        TᵀN⁻¹T = sphenical_harmonics.T0' * set_params.N⁻¹_set[i] * sphenical_harmonics.T0
        namewo = basename(mask_path)
        npzwrite("../../T_N_inv/TᵀN⁻¹_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax).npy", TᵀN⁻¹)
        npzwrite("../../T_N_inv_T/TᵀN⁻¹T_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax).npy", TᵀN⁻¹T)
    end
end