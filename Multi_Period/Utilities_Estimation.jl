#= Utilities for Estimation of Dynamic Chilhood Problem =#

using DataFrames
using KernelDensity
using LatexPrint

include("Utilities_Solution.jl")

## TESTING

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Field Paper 2/R/Output/nlsy79_multi_period_est_nonneg.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data,4)

initial_state_data = nlsy79data_formatted

## END TESTING

#= Data Manipulation =#

# transform data into format that can be read by moment generation function

function data_transform(df::DataFrame, T::Int64)

  # extract child id
  child_id = unique(df[:child_id])

  # initialize structures to store states and choices
  states = Array{Array{Float64}}(T-1)
  choices_s = Array{Array{Float64}}(T-1)
  choices_x = Array{Array{Float64}}(T-1)

  # loop through periods and store states and choices by period
  for t in 1:T-1
     states_t = zeros(length(child_id),3)
     choices_s_t = zeros(length(child_id))
     choices_x_t = zeros(length(child_id))
     for i in 1:length(child_id)
       states_t[i,1] = df[intersect(find(x->x==t,df[:period_id]),
                        find(x->x==child_id[i],df[:child_id])),:inc_period99][1]
       states_t[i,2] = df[intersect(find(x->x==t,df[:period_id]),
                        find(x->x==child_id[i],df[:child_id])),:hh_net_worth99][1]
       states_t[i,3] = df[intersect(find(x->x==t,df[:period_id]),
                        find(x->x==child_id[i],df[:child_id])),:piat_math_raw][1]
       choices_s_t[i] = df[intersect(find(x->x==t,df[:period_id]),
                        find(x->x==child_id[i],df[:child_id])),:savings_period99][1]
       choices_x_t[i] = df[intersect(find(x->x==t,df[:period_id]),
                        find(x->x==child_id[i],df[:child_id])),:home_dollar_adjust_period][1]
     end
     states[t] = states_t
     choices_s[t] = choices_s_t
     choices_x[t] = choices_x_t
   end

   # reorganize for output
   states_y = Array{Array{Float64}}(T-1)
   states_a = Array{Array{Float64}}(T-1)
   states_b = Array{Array{Float64}}(T-1)

   for t in 1:T-1
     states_y[t] = states[t][:,1]
     states_a[t] = states[t][:,2]
     states_b[t] = states[t][:,3]
   end

   return states_y, states_a, states_b, choices_s, choices_x

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
  T=4, seed=1234, S=10, N=100, sample_code="nodraw")

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
  hh_prefs = zeros(N,3)
  for n in 1:N
    hh_prefs[n,1:2] = exp.(pref_draws[n,:])./(1+exp(pref_draws[n,1])+exp(pref_draws[n,2]))
    hh_prefs[n,3] = 1/(1+exp(pref_draws[n,1])+exp(pref_draws[n,2]))
  end

  # store household shocks
  shocks_y = Array{Array{Float64}}(T-1)
  shocks_b = Array{Array{Float64}}(T-1)

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
  paramsdec::ParametersDec, paramsshock::ParametersShock, stategrids::StateGrids)

  # extract N, T, and S from initial states and shock paths
  N = length(initial_states[:,1])
  T = length(shocks_y)+1
  S = length(shocks_y[1][1,:])

  # store household state and decisions
  states_y = Array{Array{Float64}}(T)
  states_a = Array{Array{Float64}}(T)
  states_b = Array{Array{Float64}}(T)
  choices_aprime = Array{Array{Float64}}(T-1)
  choices_x = Array{Array{Float64}}(T-1)

  for t in 1:T
    # initialize states
    states_y[t] = zeros(N,S)
    states_a[t] = zeros(N,S)
    states_b[t] = zeros(N,S)

    # initialize choices
    if t <= T-1
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
    paramsdec.alpha1, paramsdec.alpha2, paramsdec.alpha3 = hh_prefs[n,:]

    # solve initial decisions and decision rules for t > t0
    V0, aprime0, x0, V_array, aprime_array, x_array = back_induction_hh!(states_y[1][n,1],
      states_a[1][n,1], states_b[1][n,1], paramsdec, paramsshock, stategrids,
      t0=1, aprime_min=1., aprime_N=10)

    # store initial decisions
    choices_aprime[1][n,:] = ones(S)*aprime0
    choices_x[1][n,:] = ones(S)*x0

    # calculate second period states given shocks
    for s in 1:S
      states_y[2][n,s] = Y_evol(states_y[1][n,s], paramsdec.rho_y, shocks_y[1][n,s])
      states_a[2][n,s] = choices_aprime[1][n,s]
      states_b[2][n,s] = HC_prod(states_b[1][n,s], choices_x[1][n,s], shocks_b[1][n,s],
        paramsdec.iota0[1], paramsdec.iota1[1], paramsdec.iota2[1], paramsdec.iota3[1])
    end

    # for household n, compute optimal decisions along each path and advance according to shocks
    for t in 2:T-1

      # identify grid on which to interpolate decision rules
      if t==2
        decision_state_grid_interp = stategrids.state_grid_interp2
      elseif t==3
        decision_state_grid_interp = stategrids.state_grid_interp3
      end

      # initialize choices for household n and period t
      choices_aprime_n_t = zeros(S)
      choices_x_n_t = zeros(S)

      # initialize next period states
      states_y_n_tplus1 = zeros(S)
      states_a_n_tplus1 = zeros(S)
      states_b_n_tplus1 = zeros(S)

      for s in 1:S
        # interpolate decision rules (impose asset lower bound of $1 and investment lower bound of $1)
        choices_aprime[t][n,s] = max(interpolate(decision_state_grid_interp, aprime_array[t],
                                  [states_y[t][n,s], states_a[t][n,s], states_b[t][n,s]]), 1.)

        choices_x[t][n,s] = max(interpolate(decision_state_grid_interp, x_array[t],
                                  [states_y[t][n,s], states_a[t][n,s], states_b[t][n,s]]), 1.)

        # calculate next period state given choice and shock
        states_y_n_tplus1[s] = Y_evol(states_y[t][n,s], paramsdec.rho_y, shocks_y[t][n,s])
        states_a_n_tplus1[s] = choices_aprime[t][n,s]
        states_b_n_tplus1[s] = HC_prod(states_b[t][n,s], choices_x[t][n,s], shocks_b[t][n,s],
          paramsdec.iota0[t], paramsdec.iota1[t], paramsdec.iota2[t], paramsdec.iota3[t])

        # correct negative y draws, bounding below at 1.0
        if states_y_n_tplus1[s] <= 0.0
          states_y_n_tplus1[s] = 1.0
        end

        # correct negative HC draws, bounding below at 1.0
        if states_b_n_tplus1[s] <= 0.0
          states_b_n_tplus1[s] = 1.0
        end

      end

      # store next period state
      states_y[t+1][n,:] = states_y_n_tplus1
      states_a[t+1][n,:] = states_a_n_tplus1
      states_b[t+1][n,:] = states_b_n_tplus1

    end

  end

  return states_y, states_a, states_b, choices_aprime, choices_x

end

function simulate_data_initial_given_S(initial_state_data,
  decision_rule_s::Array, decision_rule_x::Array,
  paramst::Parameterst; seed=1234, S=10, N=1000, sample_code="nodraw")

  # extract time horizon from decision rules
  T = length(decision_rule_s)+1

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

  # create GridInterpolations object with decision state grid across which to interpolate decision rules
  # need separate grid for each time period, determined by parameters
  decision_state_grid_interp1 = RectangleGrid(unique(paramst.state_grid1[:,1]),
                                unique(paramst.state_grid1[:,2]), unique(paramst.state_grid1[:,3]))
  decision_state_grid_interp2 = RectangleGrid(unique(paramst.state_grid2[:,1]),
                                unique(paramst.state_grid2[:,2]), unique(paramst.state_grid2[:,3]))
  decision_state_grid_interp3 = RectangleGrid(unique(paramst.state_grid3[:,1]),
                                unique(paramst.state_grid3[:,2]), unique(paramst.state_grid3[:,3]))

  # store household shocks
  shocks_y = Array{Array{Float64}}(T-1)
  shocks_b = Array{Array{Float64}}(T-1)

  # draw shocks, updating seed by +1 after every period to get different vector of draws
  # for each household and time, draw S shocks, so we have S paths per household
  for t in 1:T-1
    srand(seed); shocks_y_t = rand(paramst.eps_y_dist,N,S)
    srand(seed); shocks_b_t = rand(paramst.eps_b_dist,N,S)

    shocks_y[t] = shocks_y_t
    shocks_b[t] = shocks_b_t

    seed += 1
  end

  # store household state and decisions
  states_y = Array{Array{Float64}}(T)
  states_a = Array{Array{Float64}}(T)
  states_b = Array{Array{Float64}}(T)
  choices_s = Array{Array{Float64}}(T-1)
  choices_x = Array{Array{Float64}}(T-1)

  # store initial state for each path and household
  states_y[1] = initial_states[:,1].*ones(N,S)
  states_a[1] = initial_states[:,2].*ones(N,S)
  states_b[1] = initial_states[:,3].*ones(N,S)

  # for each time period during childhood, interpolate decisions and advance state given shocks
  # solve S times, one for each path of shocks
  for t in 1:T-1

    # identify grid on which to interpolate decision rules
    if t==1
      decision_state_grid_interp = decision_state_grid_interp1
    elseif t==2
      decision_state_grid_interp = decision_state_grid_interp2
    elseif t==3
      decision_state_grid_interp = decision_state_grid_interp3
    else
      throw(error("t=1, 2, or 3"))
    end

    # initialize choices for period t
    choices_s[t] = zeros(N,S)
    choices_x[t] = zeros(N,S)

    # initialize next period states
    states_y_tplus1 = zeros(N,S)
    states_a_tplus1 = zeros(N,S)
    states_b_tplus1 = zeros(N,S)

    for s in 1:S
      for n in 1:N
        # interpolate decision rules (impose asset lower bound of $1)
        choices_s[t][n,s] = max(interpolate(decision_state_grid_interp, decision_rule_s[t],
                                  [states_y[t][n,s], states_a[t][n,s], states_b[t][n,s]]),
                                  -1*states_a[t][n,s]*(1+paramst.r) + 1.)

        choices_x[t][n,s] = max(interpolate(decision_state_grid_interp, decision_rule_x[t],
                                  [states_y[t][n,s], states_a[t][n,s], states_b[t][n,s]]), 1.)

        # calculate next period state given choice and shock
        states_y_tplus1[n,s] = Y_evol(states_y[t][n,s], shocks_y[t][n,s], paramst)
        states_a_tplus1[n,s] = choices_s[t][n,s] + states_a[t][n,s]*(1+paramst.r)
        states_b_tplus1[n,s] = HC_prod(states_b[t][n,s], choices_x[t][n,s], shocks_b[t][n,s], t, paramst)

        # correct negative y draws, bounding below at 1.0
        for n in 1:N
          if states_y_tplus1[n,s] <= 0.0
            states_y_tplus1[n,s] = 1.0
          end
        end

        # correct negative HC draws, bounding below at 1.0
        for n in 1:N
          if states_b_tplus1[n,s] <= 0.0
            states_b_tplus1[n,s] = 1.0
          end
        end

      end
    end

    # store next period state
    states_y[t+1] = states_y_tplus1
    states_a[t+1] = states_a_tplus1
    states_b[t+1] = states_b_tplus1

  end

  return states_y, states_a, states_b, choices_s, choices_x, shocks_y, shocks_b

end

#= Moment Generation =#

# calculate moments distribution of states and choices given S paths for each household

function moment_gen_S_dist(data_formatted_S; restrict_flag=0, T=4)

  # extract T-1 given simulated data (moments only calculated for non-terminal period)
  Tminus1 = T-1

  # extract S given simulate data
  S = length(data_formatted_S[1][1][1,:])

  # extract N given simulated data
  N = length(data_formatted_S[1][1][:,1])

  # initialize storage of state and choice moments
  y_moments = Array{Array{Float64}}(Tminus1)
  a_moments = Array{Array{Float64}}(Tminus1)
  b_moments = Array{Array{Float64}}(Tminus1)
  s_moments = Array{Array{Float64}}(Tminus1)
  x_moments = Array{Array{Float64}}(Tminus1)
  y_cov = Array{Array{Float64}}(Tminus1)
  a_cov = Array{Array{Float64}}(Tminus1)
  b_cov = Array{Array{Float64}}(Tminus1)

  # calculate first through third moments of states and controls at each period
  for t in 1:Tminus1

    # initialize moment storage
    y_mom_t = zeros(2,S)
    a_mom_t = zeros(2,S)
    b_mom_t = zeros(2,S)
    s_mom_t = zeros(2,S)
    x_mom_t = zeros(2,S)
    y_cov_t = zeros(2,S)
    a_cov_t = zeros(2,S)
    b_cov_t = zeros(2,S)

    for s in 1:S

      y_mom_t[:,s] = [mean(log.(data_formatted_S[1][t][:,s])) var(log.(data_formatted_S[1][t][:,s]))^0.5]
      a_mom_t[:,s] = [mean(log.(data_formatted_S[2][t][:,s])) var(log.(data_formatted_S[2][t][:,s]))^0.5]
      b_mom_t[:,s] = [mean(data_formatted_S[3][t][:,s]) var(data_formatted_S[3][t][:,s])^0.5]
      s_mom_t[:,s] = [mean(data_formatted_S[4][t][:,s]) var(data_formatted_S[4][t][:,s])^0.5]
      x_mom_t[:,s] = [mean(data_formatted_S[5][t][:,s]) var(data_formatted_S[5][t][:,s])^0.5]
      y_cov_t[:,s] = [cov(log.(data_formatted_S[1][t][:,s]),data_formatted_S[4][t][:,s]) cov(log.(data_formatted_S[1][t][:,s]),data_formatted_S[5][t][:,s])]
      a_cov_t[:,s] = [cov(log.(data_formatted_S[2][t][:,s]),data_formatted_S[4][t][:,s]) cov(log.(data_formatted_S[2][t][:,s]),data_formatted_S[5][t][:,s])]
      b_cov_t[:,s] = [cov(log.(data_formatted_S[3][t][:,s]),data_formatted_S[4][t][:,s]) cov(log.(data_formatted_S[3][t][:,s]),data_formatted_S[5][t][:,s])]

    end

    # average moments over S paths
    y_moments[t] =  transpose(mean(y_mom_t,2))
    a_moments[t] = transpose(mean(a_mom_t,2))
    b_moments[t] = transpose(mean(b_mom_t,2))
    s_moments[t] = transpose(mean(s_mom_t,2))
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
      moment_stack  = vcat(moment_stack, s_moments[t][i])
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

  moment_restrict = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]

  # restrict moments to "relevant" moments if prompted
  if restrict_flag == 1
    moment_stack = transpose(moment_stack[moment_restrict])
  elseif restrict_flag == 0
    moment_stack = moment_stack
  else
    throw(error("restrict_flag must be 0 or 1"))
  end

  return moment_stack, y_moments, a_moments, b_moments, s_moments, x_moments, y_cov, a_cov, b_cov

end

#= Estimation Utilities =#

## Jointly Estimate All Parameters via SMM

function smm_all(data_formatted, paramst::Parameterst;
  alpha01_start=0.75, alpha1_start=0.5, B_start=10.,
  rho_y_start=1.007, eps_y_var_start=0.2513478, eps_b_var_start=0.022,
  iota01_start=1.87428633, iota02_start=1.88977362, iota03_start=0.84394021,
  iota11_start=0.42122805, iota12_start=0.43082281, iota13_start=0.64656581,
  iota21_start=0.05979631, iota22_start=0.05642741, iota23_start=0.09121635,
  iota31_start=0.0004, iota32_start=0.0024, iota33_start=0.019,
  N=1000, T=4, S=100,
  opt_code="neldermead", sample_code="nodraw", restrict_flag=0, seed=1234, error_log_flag=0)

  # extract initial states
  initial_states_data = data_formatted

  # generate data moments
  data_moments = moment_gen_S_dist(data_formatted,restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  # copy parameters structure to modify
  paramst_float = deepcopy(paramst)

  # define objective function given data moments
  smm_obj_inner(param_vec) = smm_obj(initial_states_data, data_moments, W, param_vec, paramst_float,
    N=N, T=T, restrict_flag=restrict_flag, S=S, seed=seed, sample_code=sample_code, error_log_flag=error_log_flag)

  # minimize objective
  if opt_code == "neldermead"
    smm_opt = optimize(smm_obj_inner, [alpha01_start, alpha1_start, B_start,
    rho_y_start, eps_y_var_start, eps_b_var_start,
    iota01_start, iota02_start, iota03_start,
    iota11_start, iota12_start, iota13_start,
    iota21_start, iota22_start, iota23_start,
    iota31_start, iota32_start, iota33_start], show_trace=true, iterations=5000)
  elseif opt_code == "lbfgs"
    smm_opt = optimize(smm_obj_inner, [alpha01_start, alpha1_start, B_start,
    rho_y_start, eps_y_var_start, eps_b_var_start,
    iota01_start, iota02_start, iota03_start,
    iota11_start, iota12_start, iota13_start,
    iota21_start, iota22_start, iota23_start,
    iota31_start, iota32_start, iota33_start], LBFGS())
  elseif opt_code == "simulatedannealing"
    smm_opt = optimize(smm_obj_inner, [alpha01_start, alpha1_start, B_start,
    rho_y_start, eps_y_var_start, eps_b_var_start,
    iota01_start, iota02_start, iota03_start,
    iota11_start, iota12_start, iota13_start,
    iota21_start, iota22_start, iota23_start,
    iota31_start, iota32_start, iota33_start], SimulatedAnnealing())
  else
    throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  return smm_opt

end

## SMM Objective Function, modifies parameters in place and takes target moments as argument

# note that error_log_flag!=1 does not store errors, but does allow model computation to proceed amongst errors

function smm_obj(initial_states, target_moments, W::Array, param_vec::Array, paramst::Parameterst;
  N=1000, T=4, restrict_flag=0, S=10, seed=1234, sample_code="nodraw", error_log_flag=0)

  println(param_vec)

  # set parameters according to guess, impose utility elasticities sum to 1
  paramst.alpha01 = param_vec[1]
  paramst.alpha02 = 1 - param_vec[1]
  paramst.alpha1 = param_vec[2]
  paramst.alpha2 = 1 - param_vec[2]
  paramst.B = param_vec[3]

  paramst.rho_y = param_vec[4]
  paramst.eps_y_var = param_vec[5]
  paramst.eps_b_var = param_vec[6]
  paramst.iota0[1] = param_vec[7]
  paramst.iota0[2] = param_vec[8]
  paramst.iota0[3] = param_vec[9]
  paramst.iota1[1] = param_vec[10]
  paramst.iota1[2] = param_vec[11]
  paramst.iota1[3] = param_vec[12]
  paramst.iota2[1] = param_vec[13]
  paramst.iota2[2] = param_vec[14]
  paramst.iota2[3] = param_vec[15]
  paramst.iota3[1] = param_vec[16]
  paramst.iota3[2] = param_vec[17]
  paramst.iota3[3] = param_vec[18]

  # relevant constraints
  if param_vec[1] >= 1.0 || param_vec[1] <= 0.0 || param_vec[2] >= 1.0 || param_vec[2] <= 0.0 ||
    param_vec[3] <= 0.0 || param_vec[3] <= 0.0 || param_vec[5] <= 0.0 || param_vec[6] <= 0.0
    obj = Inf
  else
    # solve model given parameters
    V_array, s_array, x_array, error_log1, error_log2 = back_induction_Optim(paramst, T, error_log_flag=error_log_flag)

    # simulate dataset
    sim_data = simulate_data_initial_given_S(initial_states, s_array, x_array,
            paramst, S=S, N=N, seed=seed, sample_code=sample_code)

    # calculate simulated data moments
    sim_moments = moment_gen_S_dist(sim_data,restrict_flag=restrict_flag)

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - target_moments[1])*W*(sim_moments[1] - target_moments[1]))[1]

  end

  return obj

end

# SMM objective function for testing, does not modify parameters in place and generates data moments

# note that error_log_flag==1 does store errors

function smm_obj_testing(data_formatted, param_vec, paramst::Parameterst;
  N=1000, T=4, restrict_flag=0, S=10, seed=1234, sample_code="nodraw", print_flag=0, error_log_flag=0)

  # initialize array that holds error logs of model computation
  error_logs1 = Any[]
  error_logs2 = Any[]

  # extract initial states
  initial_states_data = data_formatted

  # generate data moments
  data_moments = moment_gen_S_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
    W[m,m] = 1/abs(data_moments[1][m])
  end

  println(param_vec)

  # set parameters according to guess, restrict utility coefficients
  paramst_float = deepcopy(paramst)

  paramst_float.alpha01 = param_vec[1]
  paramst_float.alpha02 = 1 - param_vec[1]
  paramst_float.alpha1 = param_vec[2]
  paramst_float.alpha2 = 1 - param_vec[2]
  paramst_float.B = param_vec[3]

  paramst_float.rho_y = param_vec[4]
  paramst_float.eps_y_var = param_vec[5]
  paramst_float.eps_b_var = param_vec[6]
  paramst_float.iota0[1] = param_vec[7]
  paramst_float.iota0[2] = param_vec[8]
  paramst_float.iota0[3] = param_vec[9]
  paramst_float.iota1[1] = param_vec[10]
  paramst_float.iota1[2] = param_vec[11]
  paramst_float.iota1[3] = param_vec[12]
  paramst_float.iota2[1] = param_vec[13]
  paramst_float.iota2[2] = param_vec[14]
  paramst_float.iota2[3] = param_vec[15]
  paramst_float.iota3[1] = param_vec[16]
  paramst_float.iota3[2] = param_vec[17]
  paramst_float.iota3[3] = param_vec[18]

  # relevant constraints
  if param_vec[1] >= 1.0 || param_vec[1] <= 0.0 || param_vec[2] >= 1.0 || param_vec[2] <= 0.0 ||
    param_vec[3] <= 0.0 || param_vec[3] <= 0.0 || param_vec[5] <= 0.0 || param_vec[6] <= 0.0
    obj = Inf
  else
    # solve model given parameters
    V_array, s_array, x_array, error_log1, error_log2 = back_induction_Optim(paramst_float, T, error_log_flag=error_log_flag)

    # if prompted, store error logs
    if error_log_flag == 1
      push!(error_logs1, error_log1)
      push!(error_logs2, error_log2)
    end

    # simulate dataset
    sim_data = simulate_data_initial_given_S(initial_states_data, s_array, x_array,
            paramst_float, S=S, N=N, seed=seed, sample_code=sample_code)

    # calculate simulated data moments
    sim_moments = moment_gen_S_dist(sim_data,restrict_flag=restrict_flag)

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

  return obj, max_error_index[1], data_moments, sim_moments, sim_data, error_logs1, error_logs2

end
