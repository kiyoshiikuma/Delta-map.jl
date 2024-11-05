using PyCall
@pyimport scipy.optimize as scipy_optimize
include("original_delta_map.jl")

function iterative_minimization(set_params::SetParams, fit_params::FitParams)
    # initialize parameters
    r_pre = Inf
    r_out = 0.5
    neg2L_pre = -Inf
    neg2L_out = 1e10
    # loop until convergence
    while (-2 * neg2L_pre + 2 * neg2L_out) > 1e-2 && abs(r_pre - r_out) > 1e-5
        # fixed r, minimize chi_sq_wrapper
        function chi_sq_wrapper(params)
            if set_params.which_model == "s1"
                fit_params.beta_s = params[1]
            elseif set_params.which_model == "d1"
                fit_params.beta_d, fit_params.T_d = params[1], params[2]
            elseif set_params.which_model == "d1 and s1"
                fit_params.beta_s, fit_params.beta_d, fit_params.T_d = params[1], params[2], params[3]
            end
            return calc_chi_sq(set_params, fit_params)
        end
        if set_params.which_model == "s1"
            bounds = [(-10.0, -0.01)]
            initial_params = [fit_params.beta_s]
        elseif set_params.which_model == "d1"
            bounds = [(0.1, 10.0), (5.0, 40.0)]
            initial_params = [fit_params.beta_d, fit_params.T_d]
        elseif set_params.which_model == "d1 and s1"
            bounds = [(-10.0, -0.01), (0.1, 10.0), (5.0, 40.0)]
            initial_params = [fit_params.beta_s, fit_params.beta_d, fit_params.T_d]
        end
        result = scipy_optimize.minimize(chi_sq_wrapper, initial_params, bounds=bounds, method="L-BFGS-B")
        if set_params.which_model == "s1"
            fit_params.beta_s = result["x"][1]
        elseif set_params.which_model == "d1"
            fit_params.beta_d, fit_params.T_d = result["x"][1], result["x"][2]
        elseif set_params.which_model == "d1 and s1"
            fit_params.beta_s, fit_params.beta_d, fit_params.T_d = result["x"][1], result["x"][2], result["x"][3]
        end
        # fixed fg parameters, minimize likelihood_wrapper
        function likelihood_wrapper(r)
            fit_params.r_est = r[1]
            return calc_likelihood(set_params, fit_params)
        end
        r_result = scipy_optimize.minimize(likelihood_wrapper, [r_out], bounds=[(0.0, Inf)], method="L-BFGS-B")
        r_out = r_result["x"][1]
        # update likelihood
        r_pre = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood(set_params, fit_params)
    end
    # return r_out, beta_d, beta_s, T_d
    return r_out, fit_params.beta_d, fit_params.beta_s, fit_params.T_d
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
        make_input_map!(set_params);
        r_out, beta_d, beta_s, T_d = iterative_minimization(set_params, fit_params)
        push!(r_values, r_out)
        push!(beta_s_values, beta_s)
        push!(beta_d_values, beta_d)
        push!(T_d_values, T_d)
    end
    return r_values, beta_d_values, beta_s_values, T_d_values
end