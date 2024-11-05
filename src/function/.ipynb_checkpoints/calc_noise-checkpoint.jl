using Healpix
using StatsPlots, Distributions, Random
using LinearAlgebra
using Random

function calc_noise_map(pol_sen::Float64, nside::Int, noise_seed::Vector{Int64})
    resol_pixel = nside2resol(nside) * (60) * (180 / pi)
    sigma = pol_sen / resol_pixel
    pixel = nside2npix(nside)
    noise_map = []
    for seed in noise_seed
        Random.seed!(seed)
        push!(noise_map, rand(Normal(0, sigma), pixel))
    end
    return noise_map, sigma
end

function calc_noise_cov_mat(pol_sen::Float64, nside::Int)
    resol_pixel = nside2resol(nside) * (60) * (180 / pi)
    sigma = pol_sen / resol_pixel
    pixel = nside2npix(nside)
    noise_cov_mat = Matrix{Float64}(I, 2 * pixel, 2 * pixel) .* sigma^2
    return noise_cov_mat
end

function noise_sigma_calc(freq::Int, nside::Int, noise_seed::Vector{Int64})
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
    if haskey(pol_sen_dict, freq)
        pol_sen = pol_sen_dict[freq]
    else
        error("Please enter a valid frequency")
        return
    end
    sigma = pol_sen / resol_pixel
    pixel = hp.nside2npix(nside)
    noise_map = []
    for seed in noise_seed
        Random.seed!(seed)
        push!(noise_map, randn(pixel) .* sigma)
    end
    return noise_map, sigma, pol_sen
end

function noise_cl_th_calc(freq::Int, nside::Int)
    noise_map, sigma, pol_sen = noise_sigma_calc(freq, nside, [1, 2, 3])
    cl_noise_th_value = (π*pol_sen/10800)^2  
    l_max = 2*nside 
    l = 0 * (1 : (l_max - 1)) .+ 1
    noise_cl_th = [0; 0; l * cl_noise_th_value]
    return noise_cl_th
end