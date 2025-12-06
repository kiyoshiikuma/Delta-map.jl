# using Extended Delta-map method to calculate likelihood function
include("calc_noise.jl")
using NPZ
using PyCall
using LinearAlgebra
using SparseArrays
using Printf
@pyimport healpy as hp
@pyimport numpy as np

# ========== 型定義 ==========
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

# --- FWHM → ファイル名タグ ---
fwhm_arcmin(fwhm_rad::Float64) = fwhm_rad * 10800 / π
function fwhm_tag(fwhm_rad::Float64; decimals::Int=1)
    fmt  = @sprintf("%.*f", decimals, fwhm_arcmin(fwhm_rad))
    safe = replace(fmt, "." => "p")
    return "fwhm_$(safe)arcmin"
end

# ========== 初期化ヘルパ ==========
function set_cholesky_terms!()
    dummy_matrix = Matrix{Float64}(I, 1, 1)
    dummy_matrix_comp = Matrix{ComplexF64}(I, 1, 1)
    DᵀN⁻¹D_L = cholesky(dummy_matrix_comp)
    A_L = cholesky(dummy_matrix)
    B_L = cholesky(dummy_matrix_comp)
    return CholeskyTerms(DᵀN⁻¹D_L, A_L, B_L)
end

function set_matrix_terms!()
    empty_vector = ComplexF64[]
    empty_matrix = Array{ComplexF64, 2}(undef, 0, 0)
    return MatrixTerms(
        empty_vector,  # DᵀN⁻¹m
        empty_matrix,  # DᵀN⁻¹Dcmb
        empty_matrix,  # A
    )
end

function set_sphenical_harmonics_terms!()
    empty_matrix_comp = Array{ComplexF64, 2}(undef, 0, 0)
    empty_array_comp  = Array{ComplexF64, 3}(undef, 0, 0, 0)
    return Sphenical_Harmonics(
        empty_matrix_comp,  # T0
        empty_array_comp,   # Wlm
        empty_array_comp,   # Xlm
    )
end

# ========== 行列ユーティリティ ==========
function extract_masked_elements(set_params::SetParams, A::AbstractMatrix)
    nonzero_indices = findall(x -> x != 0, set_params.mask)
    n = length(nonzero_indices)
    masked_elements = zeros(2n, 2n)   # ← 既存スタイルに準拠（2*n）
    m_size = Int(size(A)[1] / 2)
    for (i, non_mask_i) in enumerate(nonzero_indices)
        for (j, non_mask_j) in enumerate(nonzero_indices)
            masked_elements[i, j]           = A[non_mask_i,           non_mask_j]
            masked_elements[i + n, j]       = A[non_mask_i + m_size,  non_mask_j]
            masked_elements[i, j + n]       = A[non_mask_i,           non_mask_j + m_size]
            masked_elements[i + n, j + n]   = A[non_mask_i + m_size,  non_mask_j + m_size]
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
    A = (A + A') / 2
    return cholesky(A)
end

function positive_definite_inverse(A::AbstractMatrix)
    A_cho = cholesky(A)
    uni = Matrix(I, size(A))
    return A_cho \ uni
end

function cholesky_logdet(A::Cholesky{T, S}) where {T, S <: AbstractMatrix}
    return 2 * sum(log.(diag(A.L)))
end

function extract_masked_values(set_params::SetParams, map::Vector{Float64})
    return [map[i] for i in 1:length(set_params.mask) if set_params.mask[i] == 1]
end

# ========== N⁻¹ の準備 ==========
function set_truncate_N⁻¹!(set_params::SetParams, lmin, lmax)
    N⁻¹_set = []
    art_noise_cov_mat = calc_noise_cov_mat(0.2, set_params.nside)  # artificial noise
    for nu_i in set_params.freq_bands
        freq_name = string(nu_i)
        dir_noise = "/Users/ikumakiyoshi/Library/Mobile Documents/com~apple~CloudDocs/study_fg_rm/program/julia_Delta_map/Delta_map/make_noise_covariance_matrix/smoothing_noise_cov_mat/"
        nside_name = "nside_"
        nside_n = string(set_params.nside)
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_noise, nside_name, nside_n, "_freq_", freq_name, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_pixsize25_beam_wo_wl.npy")
        noise_cov_mat = npzread(noise_cov_name)
        noise_cov_inv = positive_definite_inverse(extract_masked_elements(set_params, noise_cov_mat + art_noise_cov_mat))
        push!(N⁻¹_set, noise_cov_inv)
    end
    set_params.N⁻¹_set = N⁻¹_set
end

# ========== T0 の構築 ==========
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
    pairs = lm_list(set_params.lmax_alm, set_params.spin)
    WXmat = zeros(ComplexF64, npix, length(pairs))
    for (j, (ℓ,m)) in enumerate(pairs)
        WXmat[:, j] = get_vec_WXlm(save_WXlm, ℓ, m)
    end
    return WXmat
end

function build_T0_masked(set_params::SetParams, Wmat::Matrix{ComplexF64}, Xmat::Matrix{ComplexF64})
    @assert length(set_params.mask) == size(Wmat,1)
    mask_bool = set_params.mask .!= 0
    Wm = Wmat[mask_bool, :]
    Xm = Xmat[mask_bool, :]
    return [ Wm  Xm;
            -Xm  Wm ]
end

function set_T0_matrix(set_params::SetParams,
                       sphenical_harmonics::Sphenical_Harmonics,
                       Wmat::Matrix{ComplexF64},
                       Xmat::Matrix{ComplexF64},
                       mask_path::AbstractString)
    # T0 をセット
    T0 = build_T0_masked(set_params, Wmat, Xmat)
    sphenical_harmonics.T0 = T0
    namewo = basename(mask_path)
    npzwrite("../../T_matrix/T0_$(namewo)_nside_$(set_params.nside)_lmin_$(2)_lmax_$(set_params.lmax_alm)", T0)
    return T0
end

# ========== TᵀN⁻¹, TᵀN⁻¹T を保存（ファイル名に WX の bl 情報を付与） ==========
"""
calc_TᵀN⁻¹T_terms(set_params, sphenical_harmonics, mask_path, nside, lmin, lmax;
                   fwhm_rad_wx, fname_decimals=1,
                   outdir_TNinv="../../T_N_inv",
                   outdir_TNinvT="../../T_N_inv_T",
                   extra_tag="")

- sphenical_harmonics.T0 を使って、各周波数 i について
  TᵀN⁻¹_i = T0' * N⁻¹_i,  TᵀN⁻¹T_i = T0' * N⁻¹_i * T0 を計算・保存。
- 保存名に WX（設計行列）側の bl(FWHM) 情報を付与：
  "..._{fwhm_XXXXarcmin}_WXbl{extra_tag}.npy"
- bl 無しなら fwhm_rad_wx=0.0 を渡す。
"""
function calc_TᵀN⁻¹T_terms(set_params::SetParams,
                            sphenical_harmonics::Sphenical_Harmonics,
                            mask_path::AbstractString,
                            nside::Int, lmin::Int, lmax::Int;
                            fwhm_rad_wx::Float64,
                            fname_decimals::Int=1,
                            outdir_TNinv::AbstractString="../../T_N_inv",
                            outdir_TNinvT::AbstractString="../../T_N_inv_T",
                            extra_tag::AbstractString="")
    mkpath(outdir_TNinv)
    mkpath(outdir_TNinvT)

    namewo   = basename(mask_path)
    tag_fwhm = fwhm_tag(fwhm_rad_wx; decimals=fname_decimals)
    tag_full = "$(tag_fwhm)_WXbl" * extra_tag    # 例: "fwhm_1100p0arcmin_WXbl_polE"

    @assert !isempty(set_params.N⁻¹_set) "set_params.N⁻¹_set が未設定です"
    @assert size(sphenical_harmonics.T0,1) > 0 "sphenical_harmonics.T0 が空です"

    T0 = sphenical_harmonics.T0
    for (i, nu_i) in enumerate(set_params.freq_bands)
        Ninv    = set_params.N⁻¹_set[i]
        TtNinv  = T0' * Ninv
        TtNinvT = TtNinv * T0

        fname1 = "TᵀN⁻¹_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_$(tag_full).npy"
        fname2 = "TᵀN⁻¹T_$(namewo)_nside_$(nside)_freq_$(nu_i)_lmin_$(lmin)_lmax_$(lmax)_$(tag_full).npy"
        path1  = joinpath(outdir_TNinv,  fname1)
        path2  = joinpath(outdir_TNinvT, fname2)

        # 既存運用に合わせ、拡張子 .npy だが npzwrite を使用（NPZ.jl は .npy も書けます）
        npzwrite(path1, TtNinv)
        npzwrite(path2, TtNinvT)
    end
end
