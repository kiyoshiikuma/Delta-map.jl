using PyCall
include("extended_delta_map.jl")
@pyimport iminuit as iminuit

function iterative_minimization(set_params::SetParams, fit_params::FitParams)
    # initialize parameters
    r_pre = Inf
    r_out = 0.1
    neg2L_pre = -Inf
    neg2L_out = 1e10
    iteration = 0
    # set options for MIGRAD optimization (using iminuit)
    while !(abs(neg2L_pre - neg2L_out) <= 1e-2 && abs(r_pre - r_out) <= 1e-5) && iteration < 100
    #while iteration < 10
        iteration += 1
        # minimize chi square with fixed r
        function chi_sq_wrapper(params::Vararg{Float64})
            if set_params.which_model == "s1"
                fit_params.beta_s = params[1]
            elseif set_params.which_model == "d1"
                fit_params.beta_d, fit_params.T_d = params[1], params[2]
            elseif set_params.which_model == "d1 and s1"
                fit_params.beta_s, fit_params.beta_d, fit_params.T_d = params[1], params[2], params[3]
            end
            return calc_chi_sq(set_params, fit_params)
        end
        # set initial parameters and bounds
        if set_params.which_model == "s1"
            initial_params = [fit_params.beta_s]
            limits = [(-10.0, -0.01)]
        elseif set_params.which_model == "d1"
            initial_params = [fit_params.beta_d, fit_params.T_d]
            limits = [(0.1, 10.0), (5.0, 40.0)]
        elseif set_params.which_model == "d1 and s1"
            initial_params = [fit_params.beta_s, fit_params.beta_d, fit_params.T_d]
            limits = [(-10.0, -0.01), (0.1, 10.0), (5.0, 40.0)]
        end
        # perform MIGRAD optimization using iminuit
        minuit = iminuit.Minuit(chi_sq_wrapper, initial_params...)
        for i in 1:length(initial_params)
            minuit[:limits][i] = limits[i]
        end
        minuit[:errordef] = 1
        minuit.migrad()
        params_optimized = minuit[:values]
        if set_params.which_model == "s1"
            fit_params.beta_s = params_optimized[1]
        elseif set_params.which_model == "d1"
            fit_params.beta_d, fit_params.T_d = params_optimized[1], params_optimized[2]
        elseif set_params.which_model == "d1 and s1"
            fit_params.beta_s, fit_params.beta_d, fit_params.T_d = params_optimized[1], params_optimized[2], params_optimized[3]
        end
        # minimize likelihood with fixed fg parameters
        function likelihood_wrapper(r::Vararg{Float64})
            fit_params.r_est = r[1]
            return calc_likelihood(set_params, fit_params)
        end
        minuit_r = iminuit.Minuit(likelihood_wrapper, fit_params.r_est)
        #minuit_r[:limits] = [(0.0, Inf)]
        minuit_r[:limits] = [(0.0, 1)]
        minuit_r[:errordef] = 1
        # set previous r value because fit_params.r_est is updated by the previous optimization
        r_pre = fit_params.r_est
        minuit_r.migrad()
        # update r value
        r_out = minuit_r[:values][1]
        fit_params.r_est = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood(set_params, fit_params)
        println("Iteration $iteration: r = $r_out, Likelihood = $neg2L_out")
        println("delta_like = ", abs(neg2L_pre - neg2L_out))
        println("delta_r = ", abs(r_pre - r_out))
    end
end

function estimate_r_distribution(set_params::SetParams, fit_params::FitParams, num_seeds::Int)
    r_values = []
    beta_s_values = []
    beta_d_values = []
    T_d_values = []
    # save original parameters
    original_set_params = deepcopy(set_params)
    original_fit_params = deepcopy(fit_params)
    for seed in 1:num_seeds
        # reset parameters to original using deepcopy
        set_params = deepcopy(original_set_params)
        fit_params = deepcopy(original_fit_params)
        set_params.seed = seed
        set_m_vec!(set_params);
        iterative_minimization(set_params, fit_params)
        push!(r_values, fit_params.r_est)
        push!(beta_s_values, fit_params.beta_s)
        push!(beta_d_values, fit_params.beta_d)
        push!(T_d_values, fit_params.T_d)
    end
    return r_values, beta_s_values, beta_d_values, T_d_values
end