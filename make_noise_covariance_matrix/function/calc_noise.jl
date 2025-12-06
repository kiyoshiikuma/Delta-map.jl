using PyCall
using Healpix
using StatsPlots, Distributions, Random
using LinearAlgebra
using Random
using NPZ
include("smoothing_cov_mat_truncate_noise.jl")
@pyimport healpy as hp

# calculate noise covariance matrix
function calc_noise_cov_mat(pol_sen, nside, noise_seed)
    resol_pixel = nside2resol(nside) * (60) * (180 / pi)
    sigma = pol_sen / resol_pixel
    pixel = nside2npix(nside)
    noise_cov_mat = Matrix{Float64}(I, 2*pixel, 2*pixel) .* sigma^2
    return noise_cov_mat
end

function fwhm_pix_arcmin(nside::Int)
    θ_pix_rad = hp.nside2resol(nside)                # ピクセル角サイズ [rad]
    return θ_pix_rad * 180 / π * 60                 # → [arcmin]
end

# noise power spectrum
function noise_cl_calc(pol_sen)
    return cl_noise_th = (π*pol_sen/10800)^2 
end

# with smoothing
function noise_cov_mat_each_freq_calc_with_smoothing(freq_band, nside)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    lmax=2*nside
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        l_max = 2*nside 
        sigma, pol_sen, fwhm_eff, _ = noise_sigma_calc(freq, nside);
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (l_max - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P .* bl).^2
        N = smoothing_calc_cov_mat(noise_cl, noise_cl, nside) # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, "_noise_cov_mat_with_smoothing", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, _ = noise_sigma_calc(freq, nside);
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P .* bl).^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_ture", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with 40GHz beam =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_40GHz_beam(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        fwhm_con = 69 # 40GHz beam
        fwhm_eff = sqrt(fwhm_con^2 - fwhm_dec^2)
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P .* bl).^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_40GHz_beam", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with 40GHz beam w/o window function =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_40GHz_beam_wo_wl(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        fwhm_con = 69 # 40GHz beam
        fwhm_eff = sqrt(fwhm_con^2 - fwhm_dec^2)
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* bl.^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_40GHz_beam_wo_wl", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with w/o beam w/o window function =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_wo_beam_w_wl(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        # calculate pixel window function
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_dec * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P ./ bl).^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_wo_beam_wo_wl", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with w/o beam w/ window function =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_wo_beam_wo_wl(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_dec * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] ./ bl.^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_wo_beam_w_wl", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with pixsizee beam w/o window function =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_pixsize25_beam_wo_wl(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        fwhm_con = fwhm_pix_arcmin(nside) * 2.5 # pixsizee beam
        fwhm_eff = sqrt(fwhm_con^2 - fwhm_dec^2)
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* bl.^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_pixsize25_beam_wo_wl", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise with pixsizee beam =========#
function truncate_noise_cov_mat_each_freq_calc_with_smoothing_pixsize25_beam(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        fwhm_con = fwhm_pix_arcmin(nside) * 2.5 # pixsizee beam
        fwhm_eff = sqrt(fwhm_con^2 - fwhm_dec^2)
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_eff * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P .* bl).^2
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_with_smoothing_pixsize25_beam", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end

#======= truncate cov mat noise w/o smoothing =========#
function truncate_noise_cov_mat_each_freq_calc_without_smoothing(freq_band, nside, lmin, lmax)
    sigma_array = zeros(length(freq_band))
    npix = nside2npix(nside)
    noise_cl_list = []
    #bl_list = []
    for (i, freq) in enumerate(freq_band)
        sigma, pol_sen, fwhm_eff, fwhm_dec = noise_sigma_calc(freq, nside);
        wl = pixwin(nside, pol=true)
        wl_P = wl[2][1:lmax + 1]
        # fwhm_eff = √(σ2² - σ1²)
        bl = gaussbeam(fwhm_dec * (pi / 10800), Int(lmax), pol=true)
        noise_cl_value = noise_cl_calc(pol_sen)
        # l_max - 1
        l = 0 * (1 : (lmax - 1)) .+ 1
        # include smoothing effect
        noise_cl = [0; 0; l * noise_cl_value] .* (wl_P ./ bl).^2 # deconvだけなので、blで割る
        N = smoothing_calc_cov_mat_truncate(noise_cl, noise_cl, nside, lmin, lmax); # Nl_EE = Nl_BB
        dir_name = "smoothing_noise_cov_mat/"
        seed_n = "_noise_seed_"
        freq_n = "_freq_"
        nside_n = "nside_"
        npy = ".npy"
        lmin_n = "_lmin_"
        lmax_n = "_lmax_"
        noise_cov_name = string(dir_name, nside_n, nside, freq_n, freq, lmin_n, lmin, lmax_n, lmax, "_truncate_noise_cov_mat_without_smoothing_ture", npy)
        npzwrite(noise_cov_name, N)
        #println(noise_cl)
        push!(noise_cl_list, noise_cl)
        #push!(bl_list, bl)
    end    
    return  noise_cl_list 
end