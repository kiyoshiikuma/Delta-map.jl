using PyCall

function fwhm_pix_arcmin(nside::Int)
    θ_pix_rad = hp.nside2resol(nside)                
    return θ_pix_rad * 180 / π * 60                 
end

function smoothing_map_fwhm(input_map, input_fwhm)
    """Smoothing map with input_fwhm"""
    input_map = np.array(input_map)
    nside = hp.get_nside(input_map)
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    bl_temp = gaussbeam(input_fwhm * (pi / 10800), Int(2*nside), pol = false)
    bl_pol = gaussbeam(input_fwhm * (pi / 10800), Int(2*nside), pol = true)
    log_bl_temp = log.(bl_temp)
    log_bl_pol = log.(bl_pol)
    # smoothing the alm
    almT_sm = hp.sphtfunc.almxfl(alm[1, :], bl_temp)
    almE_sm = hp.sphtfunc.almxfl(alm[2, :], bl_pol)
    almB_sm = hp.sphtfunc.almxfl(alm[3, :], bl_pol)
    smoothed_map = hp.sphtfunc.alm2map((almT_sm, almE_sm, almB_sm), nside, lmax=2*nside, pixwin=false, verbose=false)
    return smoothed_map
end

function deconv_map_fwhm(input_map, input_fwhm)
    """Smoothing map with input_fwhm"""
    input_map = np.array(input_map)
    nside = hp.get_nside(input_map)
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    bl_temp = gaussbeam(input_fwhm * (pi / 10800), Int(2*nside), pol = false)
    bl_pol = gaussbeam(input_fwhm * (pi / 10800), Int(2*nside), pol = true)
    log_bl_temp = log.(bl_temp)
    log_bl_pol = log.(bl_pol)
    # deconvolve the alm
    almT_sm = hp.sphtfunc.almxfl(alm[1, :], exp.(-log_bl_temp))
    almE_sm = hp.sphtfunc.almxfl(alm[2, :], exp.(-log_bl_pol))
    almB_sm = hp.sphtfunc.almxfl(alm[3, :], exp.(-log_bl_pol))
    #almT_sm = hp.sphtfunc.almxfl(alm[1, :], 1/bl_temp)
    #almE_sm = hp.sphtfunc.almxfl(alm[2, :], 1/bl_pol)
    #almB_sm = hp.sphtfunc.almxfl(alm[3, :], 1/bl_pol)
    smoothed_map = hp.sphtfunc.alm2map((almT_sm, almE_sm, almB_sm), nside, lmax=2*nside, pixwin=false, verbose=false)
    return smoothed_map
end

#=================True setting map =================#
read_tqu(path) = hp.read_map(path, field=(0,1,2))  
function read_map(set_params::SetParams, freq::Int, nside_in::Int)
    base_dir = "../input_map/fg_map/"
    base_dir_cmb = "../input_map/cmb_map/"
    nside_str = "_nside_$(nside_in)"
    r_n    = string(set_params.r_input)
    seed_n = string(set_params.seed)
    GHz    = "_GHz"
    Cmb    = joinpath(base_dir_cmb, "r_$(r_n)$(nside_str)_seed_$(seed_n)")
    Synch1  = joinpath(base_dir,    "Synch_$(freq)$(GHz)$(nside_str)")
    Synch5 = joinpath(base_dir,    "Synch_s5_$(freq)$(GHz)$(nside_str).fits")
    Dust1   = joinpath(base_dir,    "Dust_$(freq)$(GHz)$(nside_str)")
    Dust10 = joinpath(base_dir,    "Dust_d10_$(freq)$(GHz)$(nside_str).fits")
    cmb  = read_tqu(Cmb)     # (3, Npix)
    s1   = read_tqu(Synch1)
    #s5   = read_tqu(Synch5)
    d1   = read_tqu(Dust1)
    #d10  = read_tqu(Dust10)
    m = if set_params.which_model == "s1"
        s1  + cmb
    elseif set_params.which_model == "s5"
        s5  + cmb
    elseif set_params.which_model == "d1"
        d1  + cmb
    elseif set_params.which_model == "d10"
        d10 + cmb
    elseif set_params.which_model == "d1 and s1"
        d1 + s1 + cmb
    else
        error("please input: 's1', 's5', 'd1', 'd10', 'd1 and s1'")
    end
    return m  
end

function read_cmb_map(set_params::SetParams, nside_in::Int)
    base_dir_cmb = "../input_map/cmb_map/"
    nside_str = "_nside_$(nside_in)"
    r_n    = string(set_params.r_input)
    seed_n = string(set_params.seed)
    GHz    = "_GHz"
    Cmb    = joinpath(base_dir_cmb, "r_$(r_n)$(nside_str)_seed_$(seed_n)")
    cmb  = read_tqu(Cmb)     # (3, Npix)
    return cmb
end

py"""
import healpy as hp
import numpy as np
def bandpass_beam_map(TQU, nside_out, lmin_out, lmax_out, conv_fwhm=None, deconv_fwhm=None):
    nside_in = hp.get_nside(TQU)
    size_out = hp.Alm.getsize(lmax_out)
    lmax_in = 2 * nside_in
    alm = hp.map2alm(TQU, lmax=lmax_in, use_pixel_weights=False)
    alm_new  = np.zeros((3,size_out), dtype=alm.dtype)
    for m in range(lmax_out+1):
        idx_start_out = hp.Alm.getidx(lmax_out, m, m)
        idx_stop_out  = hp.Alm.getidx(lmax_out, lmax_out, m)
        idx_start_in  = hp.Alm.getidx(lmax_in, m, m)
        idx_stop_in   = hp.Alm.getidx(lmax_in, lmax_out, m)
        alm_new[0,idx_start_out:idx_stop_out+1] = alm[0,idx_start_in:idx_stop_in+1]
        alm_new[1,idx_start_out:idx_stop_out+1] = alm[1,idx_start_in:idx_stop_in+1]
        alm_new[2,idx_start_out:idx_stop_out+1] = alm[2,idx_start_in:idx_stop_in+1]
    # beam convolution / deconvolution
    if conv_fwhm is not None:
        beam_conv = hp.gauss_beam(conv_fwhm * (np.pi / 10800), lmax=lmax_out, pol=True)
        bl_T = beam_conv[:, 0]
        bl_E = beam_conv[:, 1]
        bl_B = beam_conv[:, 2]
        alm_new_T_conv = hp.almxfl(alm_new[0, :], bl_T)
        alm_new_E_conv = hp.almxfl(alm_new[1, :], bl_E)
        alm_new_B_conv = hp.almxfl(alm_new[2, :], bl_B)
        alm_new = np.vstack([alm_new_T_conv, alm_new_E_conv, alm_new_B_conv])
    if deconv_fwhm is not None:
        beam_deconv = hp.gauss_beam(deconv_fwhm * (np.pi / 10800), lmax=lmax_out, pol=True)
        bl_T = beam_deconv[:, 0]
        bl_E = beam_deconv[:, 1]
        bl_B = beam_deconv[:, 2]
        # Avoid division by zero
        bl_T[bl_T == 0] = 1e-10
        bl_E[bl_E == 0] = 1e-10
        bl_B[bl_B == 0] = 1e-10
        alm_new_T_deconv = hp.almxfl(alm_new[0, :], 1.0 / bl_T)
        alm_new_E_deconv = hp.almxfl(alm_new[1, :], 1.0 / bl_E)
        alm_new_B_deconv = hp.almxfl(alm_new[2, :], 1.0 / bl_B)
        alm_new = np.vstack([alm_new_T_deconv, alm_new_E_deconv, alm_new_B_deconv])
    return hp.alm2map(alm_new, nside_out, lmax=lmax_out, pixwin=False)
    """

py"""
import healpy as hp
import numpy as np
def simple_bandpass_map(m, nside_out, lmin, lmax):
    # 1) ℓ=0…lmax の alm を取得
    alm = hp.map2alm(m, lmax=lmax, use_pixel_weights=False)
    return hp.alm2map(alm, nside_out, lmax=lmax)
    """

function set_truncate_m_vec!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Set m_vec
    """
    m_set = []
    nside_in = 128
    Random.seed!(set_params.seed)
    # Artificial noise map (0.2 μK, nside, seed)
    noise_art_cmb, sigma = calc_noise_map(0.2, set_params.nside)
    for (i, nu_i) in enumerate(set_params.freq_bands)   
        # read map
        map_cmb_fg = read_map(set_params, nu_i, nside_in);
        # noisde map calculation
        noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(nu_i, nside_in)
        # sum the maps (CMB + FG)_sm + noise
        map = smoothing_map_fwhm(map_cmb_fg, fwhm_dec) + [noise_map.T noise_map.Q noise_map.U]' 
        # bandpass + deconvolution (smoothing)
        truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
        # Artificial noise map (0.2 μK, nside, seed)
        noise_art_freq, sigma = calc_noise_map(0.2, set_params.nside)
        Q = truncate_map[2, :]
        U = truncate_map[3, :]
        masked_smoothed_Q = extract_masked_values(set_params, Q + noise_art_cmb.Q + noise_art_freq.Q)
        masked_smoothed_U = extract_masked_values(set_params, U + noise_art_cmb.U + noise_art_freq.U)
        masked_smoothed_m_vec_nu = [masked_smoothed_Q; masked_smoothed_U]       
        push!(m_set, masked_smoothed_m_vec_nu)   
    end
    set_params.m_set = m_set  
end

function make_input_map!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Make data model.
    """
    nside_in = 128
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    for nu_i in set_params.freq_bands
        # read map
        map_cmb_fg = read_map(set_params, nu_i, nside_in);
        # noisde map calculation
        noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(nu_i, nside_in)
        map = smoothing_map_fwhm(map_cmb_fg, fwhm_dec)
        truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
        Q = truncate_map[2, :]
        U = truncate_map[3, :]
        push!(Q_map, Q)
        push!(U_map, U)
    end
    set_params.Q_map = Q_map
    set_params.U_map = U_map
end

function make_input_map_w_noise!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Make data model.
    """
    nside_in = 128
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    for nu_i in set_params.freq_bands
        # read map
        map_cmb_fg = read_map(set_params, nu_i, nside_in);
        # noisde map calculation
        noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(nu_i, nside_in)
        map = smoothing_map_fwhm(map_cmb_fg, fwhm_dec) + [noise_map.T noise_map.Q noise_map.U]' 
        truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
        Q = truncate_map[2, :]
        U = truncate_map[3, :]
        push!(Q_map, Q)
        push!(U_map, U)
    end
    set_params.Q_map = Q_map
    set_params.U_map = U_map
end

function make_fg_map!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Make data model.
    """
    nside_in = 128
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    for nu_i in set_params.freq_bands
        # read map
        map_cmb_fg = read_map(set_params, nu_i, nside_in);
        # read cmb map
        map_cmb = read_cmb_map(set_params, nside_in);
        # noisde map calculation
        noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(nu_i, nside_in)
        map = map_cmb_fg - map_cmb
        # truncate + smoothing
        smoothed_map = smoothing_map_fwhm(map, fwhm_con)
        truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
        Q = truncate_map[2, :]
        U = truncate_map[3, :]
        push!(Q_map, Q)
        push!(U_map, U)
    end
    return Q_map, U_map
end

function make_noise_map!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Make data model.
    """
    nside_in = 128
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    for nu_i in set_params.freq_bands
        # noisde map calculation
        noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(nu_i, nside_in)
        map = [noise_map.T noise_map.Q noise_map.U]' 
        truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
        Q = truncate_map[2, :]
        U = truncate_map[3, :]
        push!(Q_map, Q)
        push!(U_map, U)
    end
    return Q_map, U_map
end

function make_cmb_map!(set_params::SetParams, lmin_out, lmax_out, fwhm_con)
    """
    Make data model.
    """
    nside_in = 128
    map_cmb = read_cmb_map(set_params, nside_in);
    noise_map, sigma, pol_sen, fwhm_dec = truncate_noise_sigma_calc(40, nside_in)
    map = smoothing_map_fwhm(map_cmb, fwhm_dec)
    truncate_map = py"bandpass_beam_map"(map, set_params.nside, lmin_out, lmax_out, fwhm_con, fwhm_dec)
    return truncate_map[2, :], truncate_map[3, :]
end