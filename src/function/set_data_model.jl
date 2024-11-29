function make_input_map_only_cmb!(set_params::SetParams)
    """
    Make data model.
    """
    cmb_data_Q = Vector{Vector{Float64}}()
    cmb_data_U = Vector{Vector{Float64}}()
    base_dir_cmb = "../map_file/cmb_map/"
    nside_str = "_nside_$(set_params.nside)"
    r_name = "r_"
    r_n = string(set_params.r_input)
    seed_name = "_seed_"
    seed_n = string(set_params.seed)
    GHz = "_GHz"
    # Construct file paths
    Cmb_name = joinpath(base_dir_cmb, "$(r_name)$(r_n)$(nside_str)$(seed_name)$(seed_n)")
    # Read maps using healpy
    cmb_data_Q = hp.read_map(Cmb_name, field=1)
    cmb_data_U = hp.read_map(Cmb_name, field=2)
    set_params.Q_map = cmb_data_Q
    set_params.U_map = cmb_data_U
end

function make_input_map!(set_params::SetParams)
    """
    Make data model.
    """
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    base_dir = "../map_file/fg_map/"
    base_dir_cmb = "../map_file/cmb_map/"
    nside_str = "_nside_$(set_params.nside)"
    r_name = "r_"
    r_n = string(set_params.r_input)
    seed_name = "_seed_"
    seed_n = string(set_params.seed)
    GHz = "_GHz"
    for freq in set_params.freq_bands
        # Construct file paths
        Cmb_name = joinpath(base_dir_cmb, "$(r_name)$(r_n)$(nside_str)$(seed_name)$(seed_n)")
        Synch_name = joinpath(base_dir, "Synch_$(freq)$(GHz)$(nside_str)")
        Dust_name = joinpath(base_dir, "Dust_$(freq)$(GHz)$(nside_str)")
        Dust_name_d0 = joinpath(base_dir, "Dust_d0_$(freq)$(GHz)$(nside_str)")
        # Read maps using healpy
        cmb_data_Q = hp.read_map(Cmb_name, field=1)
        cmb_data_U = hp.read_map(Cmb_name, field=2)
        synch_data_Q = hp.read_map(Synch_name, field=1)
        synch_data_U = hp.read_map(Synch_name, field=2)
        Dust_data_Q = hp.read_map(Dust_name, field=1)
        Dust_data_U = hp.read_map(Dust_name, field=2)
        #Dust_d0_data_Q = hp.read_map(Dust_name_d0, field=1)
        #Dust_d0_data_U = hp.read_map(Dust_name_d0, field=2)
        # Combine maps based on the model
        if set_params.which_model == "s1"
            m_Q = synch_data_Q + cmb_data_Q
            m_U = synch_data_U + cmb_data_U
        elseif set_params.which_model == "d1"
            m_Q = Dust_data_Q + cmb_data_Q
            m_U = Dust_data_U + cmb_data_U
        elseif set_params.which_model == "d0"
            m_Q = Dust_d0_data_Q + cmb_data_Q
            m_U = Dust_d0_data_U + cmb_data_U
        elseif set_params.which_model == "d1 and s1"
            m_Q = synch_data_Q + Dust_data_Q + cmb_data_Q
            m_U = synch_data_U + Dust_data_U + cmb_data_U
        else
            error("Invalid model specified. Please enter a valid model: 's1', 'd1', 'd0', or 'd1 and s1'.")
        end
        push!(Q_map, m_Q)
        push!(U_map, m_U)
    end
    set_params.Q_map = Q_map
    set_params.U_map = U_map
end

function make_input_map!(set_params::SetParams, cmb::Vector{Float64})
    """
    Make data model, please prepare the input cmb_map.
    """
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    base_dir = "../map_file/fg_map/"
    nside_str = "nside_$(set_params.nside)"
    GHz = "_GHz_"
    for freq in set_params.freq_bands
        # Construct file paths
        Synch_name = joinpath(base_dir, "Synch_$(freq)$(GHz)$(nside_str)")
        Dust_name = joinpath(base_dir, "Dust_$(freq)$(GHz)$(nside_str)")
        Dust_name_d0 = joinpath(base_dir, "Dust_d0_$(freq)$(GHz)$(nside_str)")
        # Read maps using healpy
        synch_data_Q = hp.read_map(Synch_name, field=1)
        synch_data_U = hp.read_map(Synch_name, field=2)
        Dust_data_Q = hp.read_map(Dust_name, field=1)
        Dust_data_U = hp.read_map(Dust_name, field=2)
        #Dust_d0_data_Q = hp.read_map(Dust_name_d0, field=1)
        #Dust_d0_data_U = hp.read_map(Dust_name_d0, field=2)
        # Combine maps based on the model
        if set_params.which_model == "s1"
            m_Q = synch_data_Q + cmb[2, :]
            m_U = synch_data_U + cmb[3, :]
        elseif set_params.which_model == "d1"
            m_Q = Dust_data_Q + cmb[2, :]
            m_U = Dust_data_U + cmb[3, :]
        elseif set_params.which_model == "d0"
            m_Q = Dust_d0_data_Q + cmb[2, :]
            m_U = Dust_d0_data_U + cmb[3, :]
        elseif set_params.which_model == "d1 and s1"
            m_Q = synch_data_Q + Dust_data_Q + cmb[2, :]
            m_U = synch_data_U + Dust_data_U + cmb[3, :]
        else
            error("Invalid model specified. Please enter a valid model: 's1', 'd1', 'd0', or 'd1 and s1'.")
        end
        push!(Q_map, m_Q)
        push!(U_map, m_U)
    end
    set_params.Q_map = Q_map
    set_params.U_map = U_map
end

function make_input_map!(set_params::SetParams, input_map_nu::Vector{Vector{Vector{Float64}}})
    """
    Make input maps, please prepare the input map (cmb + Fg)
    """
    Q_map = Vector{Vector{Float64}}()
    U_map = Vector{Vector{Float64}}()
    for (i, freq) in enumerate(set_params.freq_bands)
        if set_params.which_model == "s1" || which_model == "d0" || which_model == "d1" || which_model == "d 1 and s1"
            m_Q = input_map_nu[i][2] 
            m_U = input_map_nu[i][3]  
        else
            error("Invalid model specified. Please enter a valid model: 's1', 'd1', 'd0', or 'd1 and s1'.")
        end
        push!(Q_map, m_Q)
        push!(U_map, m_U)
    end
    set_params.Q_map = Q_map
    set_params.U_map = U_map
end

function smoothing_map_fwhm(input_map, input_fwhm, nside)
    """Smoothing map with input_fwhm"""
    input_map = np.array(input_map)
    alm = hp.sphtfunc.map2alm(input_map, lmax = 2*nside)
    smoothed_map = hp.sphtfunc.alm2map(alm, nside, lmax=2*nside, pixwin=true, verbose=false, fwhm=input_fwhm*pi/10800.)
    return smoothed_map
end

function make_data_m_nu(set_params::SetParams, freq::Int)
    N_pix = Healpix.nside2npix(set_params.nside)
    data_m = zeros(Float64, 2 * N_pix)
    base_dir = "../map_file/fg_map/"
    base_dir_cmb = "../map_file/cmb_map/"
    nside_str = "_nside_$(set_params.nside)"
    r_name = "r_"
    r_n = string(set_params.r_input)
    seed_name = "_seed_"
    seed_n = string(set_params.seed)
    GHz = "_GHz"
    Cmb_name = joinpath(base_dir_cmb, "$(r_name)$(r_n)$(nside_str)$(seed_name)$(seed_n)")
    Synch_name = joinpath(base_dir, "Synch_$(freq)$(GHz)$(nside_str)")
    Dust_name = joinpath(base_dir, "Dust_$(freq)$(GHz)$(nside_str)")
    Dust_name_d0 = joinpath(base_dir, "Dust_d0_$(freq)$(GHz)$(nside_str)")
    cmb_data_Q = hp.read_map(Cmb_name, field=1)
    cmb_data_U = hp.read_map(Cmb_name, field=2)
    synch_data_Q = hp.read_map(Synch_name, field=1)
    synch_data_U = hp.read_map(Synch_name, field=2)
    Dust_data_Q = hp.read_map(Dust_name, field=1)
    Dust_data_U = hp.read_map(Dust_name, field=2)
    #Dust_d0_data_Q = hp.read_map(Dust_name_d0, field=1)
    #Dust_d0_data_U = hp.read_map(Dust_name_d0, field=2)
    if set_params.which_model == "s1"
        m_Q = synch_data_Q + cmb_data_Q 
        m_U = synch_data_U + cmb_data_U
    elseif set_params.which_model == "d1"
        m_Q = Dust_data_Q + cmb_data_Q 
        m_U = Dust_data_U + cmb_data_U
    elseif set_params.which_model == "d0"
        m_Q = Dust_d0_data_Q + cmb_data_Q 
        m_U = Dust_d0_data_U + cmb_data_U
    elseif set_params.which_model == "d1 and s1"
        m_Q = synch_data_Q + Dust_data_Q + cmb_data_Q 
        m_U = synch_data_U + Dust_data_U + cmb_data_U 
    else
        error("正しいモデルを指定してください: 's1', 'd1', 'd0', または 'd1 and s1'")
    end
    data_m[1 : N_pix] = m_Q
    data_m[1 + N_pix : 2N_pix] = m_U
    return data_m
end

function set_m_vec!(set_params::SetParams)
    """
    Set m_vec
    """
    m_set = []
    npix_a = hp.nside2npix(set_params.nside)
    Random.seed!(set_params.seed)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        # Noise map calculation
        noise_map, sigma, pol_sen = noise_sigma_calc(nu_i, set_params.nside)
        # Artificial noise map (0.2 μK, nside, seed)
        noise_art_1, sigma = calc_noise_map(0.2, set_params.nside)
        noise_art_2, sigma = calc_noise_map(0.2, set_params.nside)
        m_vec_nu = make_data_m_nu(set_params, nu_i)
        # Need to make the input map three to expand with spin 2
        fwhm_con = 2200 * (4 / nside) ^ 2
        smoothed_map = smoothing_map_fwhm([0 * m_vec_nu[1 : npix_a], m_vec_nu[1 : npix_a], m_vec_nu[npix_a + 1 : 2npix_a]], fwhm_con, set_params.nside)
        Q = smoothed_map[2, :]
        U = smoothed_map[3, :]       
        masked_smoothed_Q = extract_masked_values(set_params, Q + noise_map.Q + noise_art_1.Q + noise_art_2.Q)
        masked_smoothed_U = extract_masked_values(set_params, U + noise_map.U + noise_art_1.U + noise_art_2.U)
        masked_smoothed_m_vec_nu = [masked_smoothed_Q; masked_smoothed_U]       
        push!(m_set, masked_smoothed_m_vec_nu)   
    end
    set_params.m_set = m_set  
end

function set_m_vec!(set_params::SetParams, input_map_nu::Vector{Matrix{Float64}})
    """
    Set m_vec 
    """
    m_set = []
    npix_a = hp.nside2npix(set_params.nside)
    Random.seed!(set_params.seed)
    for (i, nu_i) in enumerate(set_params.freq_bands)
        noise_art_1, sigma = calc_noise_map(0.2, set_params.nside)
        noise_art_2, sigma = calc_noise_map(0.2, set_params.nside)
        # Need to make the input map three to expand with spin 2
        fwhm_con = 2200 * (4 / nside) ^ 2
        smoothed_map = smoothing_map_fwhm([0 * input_map_nu[i][1, :], input_map_nu[i][2, :], input_map_nu[i][3, :]], fwhm_con, set_params.nside)
        Q = smoothed_map[2, :]
        U = smoothed_map[3, :]       
        masked_smoothed_Q = extract_masked_values(set_params, Q + noise_art_1.Q + noise_art_2.Q)
        masked_smoothed_U = extract_masked_values(set_params, U + noise_art_1.Q + noise_art_2.Q)
        masked_smoothed_m_vec_nu = [masked_smoothed_Q; masked_smoothed_U]       
        push!(m_set, masked_smoothed_m_vec_nu)   
    end
    set_params.m_set = m_set  
end

function set_m_vec_ml!(set_params::SetParams, noise)
    """
    Set m_vec
    """
    m_set = []
    npix_a = hp.nside2npix(set_params.nside)
    Random.seed!(set_params.seed)
    if noise == false
        for (i, nu_i) in enumerate(set_params.freq_bands)
            m_vec_nu = make_data_m_nu(set_params, nu_i)
            push!(m_set, m_vec_nu)
        end
    else
        for (i, nu_i) in enumerate(set_params.freq_bands)
            # Noise map calculation
            noise_map, sigma, pol_sen = noise_sigma_calc(nu_i, set_params.nside)
            # Artificial noise map (0.2 μK, nside, seed)
            noise_art_1, sigma = calc_noise_map(0.2, set_params.nside)
            noise_art_2, sigma = calc_noise_map(0.2, set_params.nside)
            noise_art_3, sigma = calc_noise_map(0.2, set_params.nside)
            m_vec_nu = make_data_m_nu(set_params, nu_i) 
            m_vec_nu_w_anoise = [m_vec_nu[1 : npix_a] + noise_map.Q + noise_art_2.Q + noise_art_3.Q; m_vec_nu[npix_a + 1 : 2npix_a] + noise_map.U + noise_art_2.U + noise_art_3.U]
            push!(m_set, m_vec_nu_w_anoise) 
        end
    end
    set_params.m_set = m_set  
end