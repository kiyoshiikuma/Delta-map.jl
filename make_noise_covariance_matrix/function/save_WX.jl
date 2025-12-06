using WignerD, Healpix, LinearAlgebra, NPZ, Printf, PyCall
@pyimport healpy as hp

fwhm_arcmin(fwhm_rad::Float64) = fwhm_rad * 10800 / π

fwhm_from_pixels(nside::Int; multiple::Real=2.5) = Healpix.nside2resol(nside) * multiple

function fwhm_tag(fwhm_rad::Float64; decimals::Int=1)
    fmt  = @sprintf("%.*f", decimals, fwhm_arcmin(fwhm_rad))
    safe = replace(fmt, "." => "p")
    return "fwhm_$(safe)arcmin"
end

function beam_vector_pol(lmax::Int, fwhm_rad::Float64)
    bl = hp.gauss_beam(fwhm_rad, lmax, pol=true)
    blT, blE, blB = bl[:, 1], bl[:, 2], bl[:, 3]
    @assert maximum(abs.(blB .- blE)) < 1e-12 "beam B is not equal to E"
    return blE
end

function calc_Yslm(nside::Int, spin::Int, lmax::Int)
    @assert lmax <= 2nside 
    npix = nside2npix(nside)
    res  = Resolution(nside)
    Y = zeros(ComplexF64, npix, lmax+1, 2lmax+1)
    ℓmin = max(abs(spin), 2)
    @inbounds @views for pix in 1:npix
        θ, φ = pix2angRing(res, pix)  # (θ, φ)
        for ℓ in ℓmin:lmax
            Base.Threads.@threads for m in -ℓ:ℓ
                Y[pix, ℓ+1, m + lmax + 1] =
                    (-1)^m * sqrt((2ℓ+1)/(4π)) * WignerD.wignerDjmn(ℓ, -m, spin, φ, θ, 0)
            end
        end
    end
    return Y
end

function calc_WX(nside::Int, lmax::Int)
    Yp = calc_Yslm(nside, +2, lmax)
    Ym = calc_Yslm(nside, -2, lmax)
    Wlm = -0.5  .* (Yp .+ Ym)
    Xlm = -0.5im .* (Yp .- Ym)
    return Wlm, Xlm
end

function make_WXmat_from_save(save_WXlm::Array{ComplexF64,3};
                              lmax::Int, b::AbstractVector{<:Real})
    npix, Ldim, _ = size(save_WXlm)
    lstore = Ldim - 1
    @assert 2 ≤ lmax ≤ lstore 
    @assert length(b) == lmax + 1 
    pairs = [(ℓ,m) for ℓ in 2:lmax for m in -ℓ:ℓ]
    M = Matrix{ComplexF64}(undef, npix, length(pairs))
    @inbounds @views for (j,(ℓ,m)) in enumerate(pairs)
        M[:, j] = b[ℓ+1] .* save_WXlm[:, ℓ+1, m + lstore + 1]
    end
    return M
end

function save_WXmat_spin(nside::Int, lmax::Int;
                         outdir::AbstractString="WX_matrix",
                         fwhm_rad::Union{Nothing,Float64}=nothing,
                         fwhm_pix_multiple::Real=2.5,
                         fname_decimals::Int=1)
    # 1) FWHM 決定（未指定ならピクセル×multiple）
    fwhm_use = isnothing(fwhm_rad) ? fwhm_from_pixels(nside; multiple=fwhm_pix_multiple) : fwhm_rad
    println("Using FWHM = $(fwhm_arcmin(fwhm_use)) arcmin\n")
    # 2) 偏光ビーム係数（長さ lmax+1）
    b = beam_vector_pol(lmax, fwhm_use)
    println("Beam b_ell (E/B) at ell=2:$(min(10,lmax)) = ", b[3:min(10,lmax)+1], "\n")

    # 3) Wlm/Xlm → WX 行列（列ごとに b_ell を適用）
    Wlm, Xlm = calc_WX(nside, lmax)
    Wmat = make_WXmat_from_save(Wlm; lmax=lmax, b=b)
    Xmat = make_WXmat_from_save(Xlm; lmax=lmax, b=b)

    # 4) 保存（あなたのパターン＋FWHMタグ）
    mkpath(outdir)
    spin = 2
    tag  = fwhm_tag(fwhm_use; decimals=fname_decimals)
    fname = "WXmat_spin_$(spin)_nside_$(nside)_lmax_$(lmax)_$(tag).npz"
    fpath = joinpath(outdir, fname)
    npzwrite(fpath, Dict("Wmat"=>Wmat, "Xmat"=>Xmat))
    return fpath
end
