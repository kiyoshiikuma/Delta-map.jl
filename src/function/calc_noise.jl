using Healpix
using StatsPlots, Distributions, Random
using LinearAlgebra
using Random

# Calculate noise map for T, Q, and U
function calc_noise_map(pol_sen::Float64, nside::Int)
    resol_pixel = nside2resol(nside) * 60 * (180 / pi)  # Pixel resolution in arcminutes
    sigma = pol_sen / resol_pixel
    npix = nside2npix(nside)
    # Generate noise maps for T, Q, U (3 x npix matrix)
    noise_maps = randn(3, npix) .* sigma
    noise_T, noise_Q, noise_U = eachrow(noise_maps)
    # Return a dictionary for T, Q, U maps
    return (T = noise_T, Q = noise_Q, U = noise_U), sigma
end

# Calculate noise covariance matrix for artificial noise
function calc_noise_cov_mat(pol_sen::Float64, nside::Int)
    resol_pixel = nside2resol(nside) * (60) * (180 / pi)
    sigma = pol_sen / resol_pixel
    pixel = nside2npix(nside)
    noise_cov_mat = Matrix{Float64}(I, 2 * pixel, 2 * pixel) .* sigma^2
    return noise_cov_mat
end

function smoothing_map_fwhm(input_map, input_fwhm, nside)
    """Smoothing map with input_fwhm"""
    input_map = np.array(input_map)
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    smoothed_map = hp.sphtfunc.alm2map(alm, nside, lmax=2*nside, pixwin=true, verbose=false, fwhm=input_fwhm*pi/10800.)
    return smoothed_map
end

# Generate noise map and sigma for a given frequency
function noise_sigma_calc(freq::Int, nside::Int)
    resol_pixel = hp.nside2resol(nside, arcmin=true)
    pol_sen_dict = Dict(
        40 => 37.42,
        50 => 33.46,
        60 => 21.31,
        68 => 16.87,
        78 => 12.07,
        89 => 11.30,
        100 => 6.56,
        119 => 4.58,
        140 => 4.79,
        166 => 5.57,
        195 => 5.85,
        235 => 10.79,
        280 => 13.80,
        337 => 21.95,
        402 => 47.45
    )
    fwhm_dict = Dict(
        40 => 69,
        50 => 56,
        60 => 48,
        68 => 43,
        78 => 39,
        89 => 35,
        100 => 29,
        119 => 25,
        140 => 23,
        166 => 21,
        195 => 20,
        235 => 19,
        280 => 24,
        337 => 20,  
        402 => 17   
    )
    if haskey(pol_sen_dict, freq)
        pol_sen = pol_sen_dict[freq]
    else
        error("Please enter a valid frequency")
        return
    end
    sigma = pol_sen / resol_pixel
    npix = hp.nside2npix(nside)
    # Generate noise maps for T, Q, and U
    noise_maps = randn(3, npix) .* sigma
    noise_T, noise_Q, noise_U = eachrow(noise_maps)
    # deconvolve and smoothing the map : (bl1 / bl2)² => exp(-σ1²)/exp(-σ2²)= exp((σ2²-σ1²))
    fwhm_dec = fwhm_dict[freq]
    fwhm_con = 2200 * (4 / nside) ^ 2
    fwhm_eff = sqrt(fwhm_con^2 - fwhm_dec^2)
    smoothed_noise_map = smoothing_map_fwhm([noise_T, noise_Q, noise_U], fwhm_eff, nside)
    noise_T, noise_Q, noise_U = eachrow(smoothed_noise_map)
    # Return as a tuple
    return (T = noise_T, Q = noise_Q, U = noise_U), sigma, pol_sen
end

# Generate theoretical noise Cl
function noise_cl_th_calc(freq::Int, nside::Int)
    _, sigma, pol_sen = noise_sigma_calc(freq, nside)
    cl_noise_th_value = (π * pol_sen / 10800)^2  
    l_max = 2 * nside 
    l = 0 * (1:(l_max - 1)) .+ 1
    noise_cl_th = [0; 0; l * cl_noise_th_value]
    return noise_cl_th
end