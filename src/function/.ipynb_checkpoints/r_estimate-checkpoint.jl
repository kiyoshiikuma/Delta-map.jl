using PyCall
using Optim
@pyimport scipy.optimize as scipy_optimize
include("extended_delta_map.jl")
@pyimport iminuit as iminuit
#==
function iterative_minimization(set_params::SetParams, fit_params::FitParams)
    # initialize parameters
    r_pre = Inf
    r_out = 0.1
    neg2L_pre = -Inf
    neg2L_out = 1e10
    iterations = Int[]
    r_values = Float64[]
    likelihood_values = Float64[]
    iteration = 0
    # set options for scipy.optimize
    options = Dict("ftol" => 1e-12, "gtol" => 1e-12, "maxiter" => 5000)
    # 収束するまでループ
    while (-2 * neg2L_pre + 2 * neg2L_out) > 1e-2 && abs(r_pre - r_out) > 1e-5
        iteration += 1
        # minimize chi square with fixed r
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
        # set initial parameters and bounds
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
        result = scipy_optimize.minimize(chi_sq_wrapper, initial_params, bounds=bounds, method="L-BFGS-B", options=options)
        if set_params.which_model == "s1"
            fit_params.beta_s = result["x"][1]
        elseif set_params.which_model == "d1"
            fit_params.beta_d, fit_params.T_d = result["x"][1], result["x"][2]
        elseif set_params.which_model == "d1 and s1"
            fit_params.beta_s, fit_params.beta_d, fit_params.T_d = result["x"][1], result["x"][2], result["x"][3]
        end
        # minimize likelihood with fixed fg parameters
        function likelihood_wrapper(r)
            fit_params.r_est = r[1]
            return (calc_likelihood(set_params, fit_params) / 1e9
        end
        r_result = scipy_optimize.minimize(likelihood_wrapper, [r_out], bounds=[(0.0, Inf)], method="L-BFGS-B", options=options)
        r_out = r_result["x"][1]
        # reset fg parameters
        r_pre = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood(set_params, fit_params)
        push!(iterations, iteration)
        push!(r_values, r_out)
        push!(likelihood_values, neg2L_out)
        println("Iteration $iteration: r = $r_out, Likelihood = $neg2L_out")
    end
    return r_out, fit_params.beta_d, fit_params.beta_s, fit_params.T_d, iterations, r_values, likelihood_values
end
==#

function iterative_minimization(set_params::SetParams, fit_params::FitParams)
    # initialize parameters
    r_pre = Inf
    r_out = 0.1
    neg2L_pre = -Inf
    neg2L_out = 1e10
    iterations = Int[]
    r_values = Float64[]
    likelihood_values = Float64[]
    iteration = 0
    # set options for MIGRAD optimization (using iminuit)
    while (-2 * neg2L_pre + 2 * neg2L_out) > 1e-2 && abs(r_pre - r_out) > 1e-5
        iteration += 1
        # minimize chi square with fixed r
        function chi_sq_wrapper(params...)
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
        function likelihood_wrapper(r...)
            fit_params.r_est = r[1]
            return calc_likelihood(set_params, fit_params)
        end
        minuit_r = iminuit.Minuit(likelihood_wrapper, r_out)
        minuit_r[:limits] = [(0.0, Inf)]
        minuit_r[:errordef] = 1
        minuit_r.migrad()
        r_out = minuit_r[:values][1]
        # reset fg parameters
        r_pre = r_out
        neg2L_pre = neg2L_out
        neg2L_out = calc_likelihood(set_params, fit_params)
        push!(iterations, iteration)
        push!(r_values, r_out)
        push!(likelihood_values, neg2L_out)
        println("Iteration $iteration: r = $r_out, Likelihood = $neg2L_out")
    end
    return r_out, fit_params.beta_d, fit_params.beta_s, fit_params.T_d, iterations, r_values, likelihood_values
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
        r_out, beta_d, beta_s, T_d = iterative_minimization(set_params, fit_params)
        push!(r_values, r_out)
        push!(beta_s_values, beta_s)
        push!(beta_d_values, beta_d)
        push!(T_d_values, T_d)
    end
    return r_values, beta_d_values, beta_s_values, T_d_values
end
