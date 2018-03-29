#= Utilities for Estimation of Dynamic Chilhood Problem =#

using DataFrames
using KernelDensity
using LatexPrint

include("Utilities_Solution_2P.jl")

## TESTING

# nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Field Paper 2/R/Output/nlsy79_multi_period_est_nonneg_alt2.csv", header=true)
#
# nlsy79data_formatted = data_transform(nlsy79data)
#
# initial_state_data = nlsy79data_formatted
#
paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()
#
# test_paths = sim_paths(initial_state_data, paramsprefs, paramsshock)
#
# test_choices = sim_choices(test_paths[1], test_paths[2], test_paths[3][1], test_paths[4][1],
#   paramsdec, paramsshock)
#
# test_mom = moment_gen_dist(test_choices)
#
# data_mom = moment_gen_dist(nlsy79data_formatted)
#
# test_params = [0.3, 0.3, 0.1, 0.1, 0.25, 0.5, 1.0, 0.997, 0.251, 0.022, 1.87, 0.42, 0.06, 0.0]
#
# test_params_prefs = [0.3, 0.3, 0.1, 0.1, 0.25, 0.5, 1.0]
#
# test_smm_obj = smm_obj_all_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock,
#   restrict_flag=1)
#
# test_smm_obj_prefs = smm_obj_prefs_testing(nlsy79data_formatted, test_params_prefs, paramsprefs, paramsdec, paramsshock,
#     restrict_flag=1)

## END TESTING

#= Data Manipulation =#

# transform data into format that can be read by moment generation function

function data_transform(df::DataFrame; T::Int64=2)

  # extract child id
  child_id = unique(df[:child_id])

  # initialize structures to store states and choices
  states = Array{Array{Float64}}(T)
  choices_aprime = Array{Array{Float64}}(T-1)
  choices_x = Array{Array{Float64}}(T-1)

  # loop through periods and store states by period
  for t in 1:T
    states_t = zeros(length(child_id),3)
    for i in 1:length(child_id)
      y_i_t = df[intersect(find(x->x==t,df[:period_id_model]),
                        find(x->x==child_id[i],df[:child_id])),:inc_period99][1]
      a_i_t = df[intersect(find(x->x==t,df[:period_id_model]),
                       find(x->x==child_id[i],df[:child_id])),:hh_net_worth99][1]
      b_i_t = df[intersect(find(x->x==t,df[:period_id_model]),
                       find(x->x==child_id[i],df[:child_id])),:piat_math_raw][1]

      if isna(y_i_t) == false
        states_t[i,1] = y_i_t
      else
        states_t[i,1] = 1.
      end
      if isna(a_i_t) == false
        states_t[i,2] = a_i_t
      else
        states_t[i,2] = -1.
      end
      if isna(b_i_t) == false
        states_t[i,3] = b_i_t
      else
        states_t[i,3] = -1.
      end

    end

    states[t] = states_t

  end

  # loop through periods and store choices by period
  for t in 1:T-1
     choices_aprime_t = zeros(length(child_id))
     choices_x_t = zeros(length(child_id))
     for i in 1:length(child_id)
       choices_aprime_t[i] = df[intersect(find(x->x==t,df[:period_id_model]),
                        find(x->x==child_id[i],df[:child_id])),:savings_period99][1]
       choices_x_t[i] = df[intersect(find(x->x==t,df[:period_id_model]),
                        find(x->x==child_id[i],df[:child_id])),:home_dollar_period_alt][1]
     end
     choices_aprime[t] = choices_aprime_t
     choices_x[t] = choices_x_t
   end

   # reorganize for output
   states_y = Array{Array{Float64}}(T)
   states_a = Array{Array{Float64}}(T)
   states_b = Array{Array{Float64}}(T)

   for t in 1:T
     states_y[t] = states[t][:,1]
     states_a[t] = states[t][:,2]
     states_b[t] = states[t][:,3]
   end

   return states_y, states_a, states_b, choices_aprime, choices_x

end

#= Parameters Structures =#

mutable struct ParametersPrefs
  mu :: Array{Float64} ## mean of preference distribution
  sigma1 :: Float64 ## std dev of alpha1
  sigma2 :: Float64 ## std dev of alpha2
  rho12 :: Float64 ## correlation between alpha1 and alpha2
  Sigma :: Array{Float64} ## covariance matrix of preference distribution

  function ParametersPrefs(;mu=[0.3, 0.3], sigma1=0.1, sigma2=0.1, rho12=0.25)

    Sigma = [sigma1^2 rho12*sigma1*sigma2; rho12*sigma1*sigma2 sigma2^2]

    new(mu, sigma1, sigma2, rho12, Sigma)

  end

end

#= DGP Simulation =#

## For Each HH, Draw Preference Vector and Simulate S Paths of Shocks, Initial Conditions Given or Drawn Non-Parametrically from Data

function sim_paths(initial_state_data, paramsprefs::ParametersPrefs, paramsshock::ParametersShock;
  T=2, seed=1234, S=10, N=100, sample_code="nodraw")

  # draw initial states and extract N if not sampling
  if sample_code=="nodraw"

    initial_states = [initial_state_data[1][1] initial_state_data[2][1] initial_state_data[3][1]]

    N = length(initial_state_data[1][1])

  elseif sample_code=="draw"

    srand(seed); initial_draws = rand(DiscreteUniform(1,length(initial_state_data[1][1])),N)

    initial_states = [initial_state_data[1][1][initial_draws] initial_state_data[2][1][initial_draws] initial_state_data[3][1][initial_draws]]
  else
    throw(error("sample_code must be nodraw or draw"))
  end

  # draw preferences for each household
  srand(seed); pref_draws = transpose(rand(MvNormal(paramsprefs.mu, paramsprefs.Sigma),N))

  # transform preferences to Cobb-Douglas
  ## NOTE: ignore utility elasticity for leisure, since there is no leisure choice yet
  hh_prefs = zeros(N,3)
  for n in 1:N
    hh_prefs[n,1:2] = exp.(pref_draws[n,:])./(exp(pref_draws[n,1])+exp(pref_draws[n,2]))
    # hh_prefs[n,3] = 1/(1+exp(pref_draws[n,1])+exp(pref_draws[n,2]))
    hh_prefs[n,3] = 0.5
  end

  # store household shocks
  shocks_y = Array{Array{Float64}}(1)
  shocks_b = Array{Array{Float64}}(1)

  # draw shocks, updating seed by +1 after every period to get different vector of draws
  # for each household and time, draw S shocks, so we have S paths per household
  for t in 1:T-1
    srand(seed); shocks_y_t = rand(paramsshock.eps_y_dist,N,S)
    srand(seed); shocks_b_t = rand(paramsshock.eps_b_dist,N,S)

    shocks_y[t] = shocks_y_t
    shocks_b[t] = shocks_b_t

    seed += 1
  end

  return initial_states, hh_prefs, shocks_y, shocks_b

end

## Given S Paths, Solve HH Problem and Store States

function sim_choices(initial_states::Array{Float64}, hh_prefs::Array{Float64},
  shocks_y::Array{Float64}, shocks_b::Array{Float64},
  paramsdec::ParametersDec, paramsshock::ParametersShock)

  # extract N, T, and S from initial states and shock paths
  N = length(initial_states[:,1])
  S = length(shocks_y[1,:])

  # store household state and decisions
  states_y = Array{Array{Float64}}(2)
  states_a = Array{Array{Float64}}(2)
  states_b = Array{Array{Float64}}(2)
  choices_aprime = Array{Array{Float64}}(1)
  choices_x = Array{Array{Float64}}(1)

  for t in 1:2
    # initialize states
    states_y[t] = zeros(N,S)
    states_a[t] = zeros(N,S)
    states_b[t] = zeros(N,S)

    # initialize choices
    if t == 1
      choices_aprime[t] = zeros(N,S)
      choices_x[t] = zeros(N,S)
    end
  end

  # store initial state for each path and household
  states_y[1] = initial_states[:,1].*ones(N,S)
  states_a[1] = initial_states[:,2].*ones(N,S)
  states_b[1] = initial_states[:,3].*ones(N,S)

  # for each household, compute decision rules given preferences and solve household problem along each path of shocks
  # initial decision is explicitly solved for, and then decision rules are interpolated on for t > t0
  for n in 1:N
    # update preferences
    ## NOTE: assuming that preferences in terminal period are same as in childhood period (no discrete choice problem)
    paramsdec.alpha1, paramsdec.alpha2, paramsdec.alpha3 = hh_prefs[n,:]

    paramsdec.alphaT1, paramsdec.alphaT2 = hh_prefs[n,1:2]

    # solve childhood period decisions
    V0, choices0 = bellman_optim_child!(states_y[1][n,1],
      states_a[1][n,1], states_b[1][n,1], paramsdec, paramsshock)

    # store childhood period decisions
    choices_aprime[1][n,:] = ones(S)*choices0[1]
    choices_x[1][n,:] = ones(S)*choices0[2]

    # calculate terminal period states given shocks
    for s in 1:S
      states_y[2][n,s] = Y_evol(states_y[1][n,s], paramsdec.rho_y, shocks_y[n,s])
      states_a[2][n,s] = choices_aprime[1][n,s]
      states_b[2][n,s] = HC_prod(states_b[1][n,s], choices_x[1][n,s], shocks_b[n,s],
        paramsdec.iota0[1], paramsdec.iota1[1], paramsdec.iota2[1], paramsdec.iota3[1])
    end

  end

  return states_y, states_a, states_b, choices_aprime, choices_x

end

#= Moment Generation =#

# calculate moments distribution of states and choices given S paths for each household

function moment_gen_dist(sim_choices; restrict_flag=0)

  # extract T-1 given simulated data (moments only calculated for non-terminal period)
  Tminus1 = length(sim_choices[4])

  # extract S given simulate data
  S = length(sim_choices[1][1][1,:])

  # extract N given simulated data
  N = length(sim_choices[1][1][:,1])

  # initialize storage of state and choice moments
  y_moments = Array{Array{Float64}}(Tminus1)
  a_moments = Array{Array{Float64}}(Tminus1)
  b_moments = Array{Array{Float64}}(Tminus1)
  aprime_moments = Array{Array{Float64}}(Tminus1)
  x_moments = Array{Array{Float64}}(Tminus1)
  y_cov = Array{Array{Float64}}(Tminus1)
  a_cov = Array{Array{Float64}}(Tminus1)
  b_cov = Array{Array{Float64}}(Tminus1)

  # calculate first and second moments of states and controls at each period
  # note that since initial conditions are taken from data, states are advanced one period
  for t in 1:Tminus1

    # initialize moment storage
    y_mom_t = zeros(2,S)
    a_mom_t = zeros(2,S)
    b_mom_t = zeros(2,S)
    aprime_mom_t = zeros(2,S)
    x_mom_t = zeros(2,S)
    y_cov_t = zeros(2,S)
    a_cov_t = zeros(2,S)
    b_cov_t = zeros(2,S)

    for s in 1:S

      y_mom_t[:,s] = [mean(log.(sim_choices[1][t+1][:,s])) var(log.(sim_choices[1][t+1][:,s]))^0.5]
      a_mom_t[:,s] = [mean(log.(sim_choices[2][t+1][:,s])) var(log.(sim_choices[2][t+1][:,s]))^0.5]
      b_mom_t[:,s] = [mean(sim_choices[3][t+1][:,s]) var(sim_choices[3][t+1][:,s])^0.5]
      aprime_mom_t[:,s] = [mean(sim_choices[4][t][:,s]) var(sim_choices[4][t][:,s])^0.5]
      x_mom_t[:,s] = [mean(sim_choices[5][t][:,s]) var(sim_choices[5][t][:,s])^0.5]
      y_cov_t[:,s] = [cov(log.(sim_choices[1][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[1][t][:,s]),sim_choices[5][t][:,s])]
      a_cov_t[:,s] = [cov(log.(sim_choices[2][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[2][t][:,s]),sim_choices[5][t][:,s])]
      b_cov_t[:,s] = [cov(log.(sim_choices[3][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[3][t][:,s]),sim_choices[5][t][:,s])]

    end

    # average moments over S paths
    y_moments[t] =  transpose(mean(y_mom_t,2))
    a_moments[t] = transpose(mean(a_mom_t,2))
    b_moments[t] = transpose(mean(b_mom_t,2))
    aprime_moments[t] = transpose(mean(aprime_mom_t,2))
    x_moments[t] = transpose(mean(x_mom_t,2))
    y_cov[t] =  transpose(mean(y_cov_t,2))
    a_cov[t] = transpose(mean(a_cov_t,2))
    b_cov[t] = transpose(mean(b_cov_t,2))

  end

  # stack moments and restrict to relevant if flagged
  moment_stack = 0.
  for i in 1:2
    for t in 1:Tminus1
      moment_stack = vcat(moment_stack, y_moments[t][i])
    end
    for t in 1:Tminus1
      moment_stack = vcat(moment_stack, a_moments[t][i])
    end
    for t in 1:Tminus1
      moment_stack  = vcat(moment_stack, b_moments[t][i])
    end
    for t in 1:Tminus1
      moment_stack  = vcat(moment_stack, aprime_moments[t][i])
    end
    for t in 1:Tminus1
      moment_stack  = vcat(moment_stack, x_moments[t][i])
    end
  end
  for i in 1:2
    for t in 1:Tminus1
      moment_stack = vcat(moment_stack, y_cov[t][i])
    end
    for t in 1:Tminus1
      moment_stack = vcat(moment_stack, a_cov[t][i])
    end
    for t in 1:Tminus1
      moment_stack  = vcat(moment_stack, b_cov[t][i])
    end
  end
  moment_stack = moment_stack[2:length(moment_stack)]

  moment_restrict = [2 3 4 5 7 8 9 10 11 12 13 14 15 16]

  # restrict moments to "relevant" moments if prompted
  if restrict_flag == 1
    moment_stack = transpose(moment_stack[moment_restrict])
  elseif restrict_flag == 0
    moment_stack = moment_stack
  else
    throw(error("restrict_flag must be 0 or 1"))
  end

  return moment_stack, y_moments, a_moments, b_moments, aprime_moments, x_moments, y_cov, a_cov, b_cov

end

#= Estimation Utilities =#

## SMM Optimization Functions

# jointly estimate all parameters via SMM

function smm_all(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  mu1_start=0.3, mu2_start=0.3, sigam1_start=0.1, sigma2_start=0.1, rho12_start=0.25,
  alphaT1_start=0.5, B_start=1.0, rho_y_start=1.007, eps_y_var_start=0.25, eps_b_var_start=0.022,
  iota0_start=1.87, iota1_start=0.42, iota2_start=0.06, iota3_start=0.0,
  N=100, T=2, S=100,
  opt_code="neldermead", sample_code="nodraw", restrict_flag=0, seed=1234, error_log_flag=0,
  opt_trace=false, opt_iter=1000)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  # define objective function given data moments
  smm_obj_inner(param_vec) = smm_obj_all(data_formatted, data_moments, W, param_vec,
    paramsprefs_float, paramsdec_float, paramsshock_float,
    N=N, T=T, restrict_flag=restrict_flag, S=S, seed=seed, sample_code=sample_code, error_log_flag=error_log_flag)

  # minimize objective
  if opt_code == "neldermead"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start, rho_y_start, eps_y_var_start, eps_b_var_start,
      iota0_start, iota1_start, iota2_start, iota3_start], show_trace=opt_trace, iterations=opt_iter)
  elseif opt_code == "lbfgs"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start, rho_y_start, eps_y_var_start, eps_b_var_start,
      iota0_start, iota1_start, iota2_start, iota3_start], LBFGS())
  elseif opt_code == "simulatedannealing"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start, rho_y_start, eps_y_var_start, eps_b_var_start,
      iota0_start, iota1_start, iota2_start, iota3_start], SimulatedAnnealing())
  else
    throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  return smm_opt

end

# estimate preference parameters only

function smm_prefs(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  mu1_start=0.3, mu2_start=0.3, sigam1_start=0.1, sigma2_start=0.1, rho12_start=0.25,
  alphaT1_start=0.5, B_start=1.0,
  N=100, T=2, S=100,
  opt_code="neldermead", sample_code="nodraw", restrict_flag=0, seed=1234, error_log_flag=0,
  opt_trace=false, opt_iter=1000)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  # define objective function given data moments
  smm_obj_inner(param_vec) = smm_obj_prefs(data_formatted, data_moments, W, param_vec,
    paramsprefs_float, paramsdec_float, paramsshock_float,
    N=N, T=T, restrict_flag=restrict_flag, S=S, seed=seed, sample_code=sample_code, error_log_flag=error_log_flag)

  # minimize objective
  if opt_code == "neldermead"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start], show_trace=opt_trace, iterations=opt_iter)
  elseif opt_code == "lbfgs"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start], LBFGS())
  elseif opt_code == "simulatedannealing"
    smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
      alphaT1_start, B_start], SimulatedAnnealing())
  else
    throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  return smm_opt

end

## Objecive Function for SMM Optimization, modifies parameters in place and takes target moments as argument

# jointly estimate all parameters

function smm_obj_all(initial_state_data, target_moments, W::Array, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=0, S=100, seed=1234, sample_code="nodraw", error_log_flag=0,
  print_flag=false)

  println(param_vec)

  # # set parameters according to guess
  # paramsprefs.mu[1], paramsprefs.mu[2], paramsprefs.sigma1, paramsprefs.sigma2,
  #   paramsprefs.rho12, paramsdec.alphaT1, paramsdec.B,
  #   paramsdec.rho_y, paramsshock.eps_y_var, paramsshock.eps_b_var,
  #   paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3 = param_vec
  #
  # # impose CES on terminal utility elasticities
  # paramsdec.alphaT2 = 1 - paramsdec.alphaT1

  # set parameters according to guess
  paramsprefs.mu[1], paramsprefs.mu[2], paramsprefs.sigma1, paramsprefs.sigma2,
    paramsprefs.rho12, paramsdec.B,
    paramsdec.rho_y, paramsshock.eps_y_var, paramsshock.eps_b_var,
    paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3 = param_vec

  # impose CES on terminal utility elasticities
  paramsdec.alphaT2 = 1 - paramsdec.alphaT1

  # relevant constraints
  if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] <= 0. || param_vec[4] <= 0. ||
    param_vec[5] < -1. || param_vec[5] > 1. || param_vec[6] <= 0. || param_vec[6] >= 1. ||
    param_vec[7] <= 0. || param_vec[8] <= 0. || param_vec[9] <= 0. || param_vec[10] <= 0.
     obj = Inf
  else
    # simulate dataset
    sim_shocks = sim_paths(initial_state_data, paramsprefs, paramsshock,
            T=T, seed=seed, N=N, S=S, sample_code=sample_code)

    sim_data = sim_choices(sim_shocks[1], sim_shocks[2], sim_shocks[3][1], sim_shocks[4][1],
      paramsdec, paramsshock)

    # calculate simulated data moments
    sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - target_moments[1])*W*(sim_moments[1] - target_moments[1]))[1]

  end

  if print_flag == true
    println(obj)
  end

  return obj

end

# estimate preference parameters only

function smm_obj_prefs(initial_state_data, target_moments, W::Array, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=0, S=100, seed=1234, sample_code="nodraw", error_log_flag=0,
  print_flag=false)

  println(param_vec)

  # set parameters according to guess
  paramsprefs.mu[1], paramsprefs.mu[2], paramsprefs.sigma1, paramsprefs.sigma2,
    paramsprefs.rho12, paramsdec.alphaT1, paramsdec.B = param_vec

  # impose CES on terminal utility elasticities
  paramsdec.alphaT2 = 1 - paramsdec.alphaT1

  # relevant constraints
  if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] <= 0. || param_vec[4] <= 0. ||
    param_vec[5] < -1. || param_vec[5] > 1. || param_vec[6] <= 0. || param_vec[6] >= 1. ||
    param_vec[7] <= 0.
     obj = Inf
  else
    # simulate dataset
    sim_shocks = sim_paths(initial_state_data, paramsprefs, paramsshock,
            T=T, seed=seed, N=N, S=S, sample_code=sample_code)

    sim_data = sim_choices(sim_shocks[1], sim_shocks[2], sim_shocks[3][1], sim_shocks[4][1],
      paramsdec, paramsshock)

    # calculate simulated data moments
    sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - target_moments[1])*W*(sim_moments[1] - target_moments[1]))[1]

  end

  if print_flag == true
    println(obj)
  end

  return obj

end

## Objecive Function for SMM, does not modify parameters in place

# jointly estimate all parameters

function smm_obj_all_testing(data_formatted, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=0, S=100, seed=1234, sample_code="nodraw", error_log_flag=0)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  println(param_vec)

  # set parameters according to guess
  paramsprefs_float.mu[1], paramsprefs_float.mu[2], paramsprefs_float.sigma1, paramsprefs_float.sigma2,
    paramsprefs_float.rho12, paramsdec_float.alphaT1, paramsdec_float.B,
    paramsdec_float.rho_y, paramsshock_float.eps_y_var, paramsshock_float.eps_b_var,
    paramsdec_float.iota0, paramsdec_float.iota1, paramsdec_float.iota2, paramsdec_float.iota3 = param_vec

  # impose CES on terminal utility elasticities
  paramsdec_float.alphaT2 = 1 - paramsdec_float.alphaT1

  # relevant constraints
  if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] <= 0. || param_vec[4] <= 0. ||
    param_vec[5] < -1. || param_vec[5] > 1. || param_vec[6] <= 0. || param_vec[6] >= 1. ||
    param_vec[7] <= 0. || param_vec[8] <= 0. || param_vec[9] <= 0. || param_vec[10] <= 0.
     obj = Inf
  else
    # simulate dataset
    sim_shocks = sim_paths(initial_state_data, paramsprefs_float, paramsshock_float,
            T=T, seed=seed, N=N, S=S, sample_code=sample_code)

    sim_data = sim_choices(sim_shocks[1], sim_shocks[2], sim_shocks[3][1], sim_shocks[4][1],
      paramsdec_float, paramsshock_float)

    # calculate simulated data moments
    sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - data_moments[1])*W*(sim_moments[1] - data_moments[1]))[1]

  end

  # find moment that results in maximum error
  if isnan(maximum(sim_moments[1] - data_moments[1]))==false
    max_error_index = find(x->x==maximum(sim_moments[1] - data_moments[1]),sim_moments[1] - data_moments[1])
  else
    max_error_index = NaN
  end

  # create table for LaTeX
  # if print_flag==1
  #   collabels_calibrated = ["Moment","Data","Model"]
  #
  #   table_calibrated  = Array(Any,(46,3))
  #   table_calibrated[1,1:3] = collabels_calibrated
  #   table_calibrated[2:46,1] = ["Mean ln y, t=1", "Mean ln y, t=2", "Mean ln y, t=3",
  #       "Mean ln a, t=1", "Mean ln a, t=2", "Mean ln a, t=3", "Mean b, t=1", "Mean b, t=2", "Mean b, t=3",
  #       "Mean s, t=1", "Mean s, t=2", "Mean s, t=3", "Mean x, t=1", "Mean x, t=2", "Mean x, t=3",
  #       "SD ln y, t=1", "SD ln y, t=2", "SD ln y, t=3",
  #           "SD ln a, t=1", "SD ln a, t=2", "SD ln a, t=3", "SD b, t=1", "SD b, t=2", "SD b, t=3",
  #           "SD s, t=1", "SD s, t=2", "SD s, t=3", "SD x, t=1", "SD x, t=2", "SD x, t=3",
  #       "Skew ln y, t=1", "Skew ln y, t=2", "Skew ln y, t=3",
  #           "Skew ln a, t=1", "Skew ln a, t=2", "Skew ln a, t=3", "Skew b, t=1", "Skew b, t=2", "Skew b, t=3",
  #           "Skew s, t=1", "Skew s, t=2", "Skew s, t=3", "Skew x, t=1", "Skew x, t=2", "Skew x, t=3"]
  #   table_calibrated[2:46,2] = round(data_moments[1],3)
  #   table_calibrated[2:46,3] = round(sim_moments[1],3)
  #
  #   tabular(table_calibrated)
  # end

  return obj, max_error_index[1], data_moments, sim_moments, sim_data

end

# estimate preference parameters only

function smm_obj_prefs_testing(data_formatted, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=0, S=100, seed=1234, sample_code="nodraw", error_log_flag=0)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  println(param_vec)

  # set parameters according to guess
  paramsprefs_float.mu[1], paramsprefs_float.mu[2], paramsprefs_float.sigma1, paramsprefs_float.sigma2,
    paramsprefs_float.rho12, paramsdec_float.alphaT1, paramsdec_float.B = param_vec

  # impose CES on terminal utility elasticities
  paramsdec_float.alphaT2 = 1 - paramsdec_float.alphaT1

  # relevant constraints
  if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] <= 0. || param_vec[4] <= 0. ||
    param_vec[5] < -1. || param_vec[5] > 1. || param_vec[6] <= 0. || param_vec[6] >= 1. ||
    param_vec[7] <= 0.
     obj = Inf
  else
    # simulate dataset
    sim_shocks = sim_paths(initial_state_data, paramsprefs_float, paramsshock_float,
            T=T, seed=seed, N=N, S=S, sample_code=sample_code)

    sim_data = sim_choices(sim_shocks[1], sim_shocks[2], sim_shocks[3][1], sim_shocks[4][1],
      paramsdec_float, paramsshock_float)

    # calculate simulated data moments
    sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - data_moments[1])*W*(sim_moments[1] - data_moments[1]))[1]

  end

  # find moment that results in maximum error
  if isnan(maximum(sim_moments[1] - data_moments[1]))==false
    max_error_index = find(x->x==maximum(sim_moments[1] - data_moments[1]),sim_moments[1] - data_moments[1])
  else
    max_error_index = NaN
  end

  # create table for LaTeX
  # if print_flag==1
  #   collabels_calibrated = ["Moment","Data","Model"]
  #
  #   table_calibrated  = Array(Any,(46,3))
  #   table_calibrated[1,1:3] = collabels_calibrated
  #   table_calibrated[2:46,1] = ["Mean ln y, t=1", "Mean ln y, t=2", "Mean ln y, t=3",
  #       "Mean ln a, t=1", "Mean ln a, t=2", "Mean ln a, t=3", "Mean b, t=1", "Mean b, t=2", "Mean b, t=3",
  #       "Mean s, t=1", "Mean s, t=2", "Mean s, t=3", "Mean x, t=1", "Mean x, t=2", "Mean x, t=3",
  #       "SD ln y, t=1", "SD ln y, t=2", "SD ln y, t=3",
  #           "SD ln a, t=1", "SD ln a, t=2", "SD ln a, t=3", "SD b, t=1", "SD b, t=2", "SD b, t=3",
  #           "SD s, t=1", "SD s, t=2", "SD s, t=3", "SD x, t=1", "SD x, t=2", "SD x, t=3",
  #       "Skew ln y, t=1", "Skew ln y, t=2", "Skew ln y, t=3",
  #           "Skew ln a, t=1", "Skew ln a, t=2", "Skew ln a, t=3", "Skew b, t=1", "Skew b, t=2", "Skew b, t=3",
  #           "Skew s, t=1", "Skew s, t=2", "Skew s, t=3", "Skew x, t=1", "Skew x, t=2", "Skew x, t=3"]
  #   table_calibrated[2:46,2] = round(data_moments[1],3)
  #   table_calibrated[2:46,3] = round(sim_moments[1],3)
  #
  #   tabular(table_calibrated)
  # end

  return obj, max_error_index[1], data_moments, sim_moments, sim_data

end

#= Informative Moment Graphing =#

# function vary_param(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64, param_name::String)
#
#   # create parameter grid
#   param_grid = linspace(param_lower, param_upper, param_N)
#
#   # create object to store moments at each parameter value
#   moment_storage = Array{Array{Float64}}(param_N)
#
#   # copy parameters structure to modify
#   paramsprefs_float = deepcopy(paramsprefs)
#   paramsdec_float = deepcopy(paramsdec)
#   paramsshock_float = deepcopy(paramsshock)
#
#   # for each parameter guess, solve model and calculate full set of moments and store
#   for n in 1:param_N
#
#   end
#
# end

#= Optimizer Testing =#

function rosen(x)

  out = (1. - x[1])^2 + 100.*(x[2] - x[1]^2)^2

  return out

end

simplex_test = optimize(rosen, [0., 0.], show_trace=true)

sa_test = optimize(rosen, [0., 0.], SimulatedAnnealing(), iterations=2000)


elseif opt_code == "lbfgs"
  smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
    alphaT1_start, B_start], LBFGS())
elseif opt_code == "simulatedannealing"
  smm_opt = optimize(smm_obj_inner, [mu1_start, mu2_start, sigam1_start, sigma2_start, rho12_start,
    alphaT1_start, B_start], SimulatedAnnealing())
else
  throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
end
