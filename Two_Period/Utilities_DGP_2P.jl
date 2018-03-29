#= Utilities for Simulating from DGP =#

println("dgp utilities loading")

# include("/home/m/mcmurry2/Utilities_Solution_2P.jl")

include("Utilities_Solution_2P.jl")

#= Structures =#

mutable struct ParametersPrefs
  mu :: Array{Float64} ## mean of terminal period preference distribution
  sigma1 :: Float64 ## std dev of alpha1
  sigma2 :: Float64 ## std dev of alpha2
  rho12 :: Float64 ## correlation between alpha1 and alpha2
  Sigma :: Array{Float64} ## covariance matrix of preference distribution

  function ParametersPrefs(;mu=[0.5, 0.5], sigma1=0.1, sigma2=0.1, rho12=0.25)

    Sigma = [sigma1^2 rho12*sigma1*sigma2; rho12*sigma1*sigma2 sigma2^2]

    new(mu, sigma1, sigma2, rho12, Sigma)

  end

end

mutable struct SimChoiceArg
   initial_states::Array{Float64}
   hh_prefs::Array{Float64}
   shocks_y::Array{Float64}
   shocks_b::Array{Float64}
   paramsdec::ParametersDec
   paramsshock::ParametersShock

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
  ## NOTE: B is calculated as sum of alpha's
  hh_prefs = zeros(N,3)
  for n in 1:N
    hh_prefs[n,1:2] = exp.(pref_draws[n,:])./(exp(pref_draws[n,1])+exp(pref_draws[n,2]))
    hh_prefs[n,3] = exp(pref_draws[n,1])+exp(pref_draws[n,2])
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

# divide paths up into N sections for parallel processing

function sim_paths_split(initial_states, hh_prefs, shocks_y, shocks_b, paramsdec::ParametersDec, paramsshock::ParametersShock;
   par_N=2)

   # initialize array of simulated choice sections
   sim_choices_arg_array = Array{SimChoiceArg}(par_N)

   # extract sample size and calculate subset length
   N = length(initial_states[:,1])
   subset_length = Int(ceil(N/par_N))

   # initialize indices for splitting
   index_start = 1
   index_end = subset_length

   for i in 1:par_N
      # subset households and paths
      initial_states_section = initial_states[index_start:index_end,:]
      hh_prefs_section = hh_prefs[index_start:index_end,:]
      shocks_y_section = shocks_y[index_start:index_end,:]
      shocks_b_section = shocks_b[index_start:index_end,:]

      # create and store simulated choice argument type with subset paths
      sim_choices_arg_section = SimChoiceArg(initial_states_section, hh_prefs_section, shocks_y_section, shocks_b_section, paramsdec, paramsshock)
      sim_choices_arg_array[i] = sim_choices_arg_section

      # update indices
      index_start = index_end + 1
      index_end = min(index_start + subset_length, N)
   end

   return sim_choices_arg_array

end

## Given S Paths, Solve HH Problem and Store States

# takes multiple arguments

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
  choices_savings = Array{Array{Float64}}(1)
  choices_x = Array{Array{Float64}}(1)

  for t in 1:2
    # initialize states
    states_y[t] = zeros(N,S)
    states_a[t] = zeros(N,S)
    states_b[t] = zeros(N,S)

    # initialize choices
    if t == 1
      choices_savings[t] = zeros(N,S)
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
    paramsdec.alphaT1, paramsdec.alphaT2, paramsdec.B = hh_prefs[n,:]

    # solve childhood period decisions
    V0, choices0 = bellman_optim_child!(states_y[1][n,1],
      states_a[1][n,1], states_b[1][n,1], paramsdec, paramsshock)

    # store childhood period decisions
    choices_savings[1][n,:] = ones(S)*choices0[1] - ones(S)*states_a[1][n,1]*(1+paramsdec.r)
    choices_x[1][n,:] = ones(S)*choices0[2]

    # calculate terminal period states given shocks
    for s in 1:S
      states_y[2][n,s] = Y_evol(states_y[1][n,s], paramsdec.rho_y, shocks_y[n,s])
      states_a[2][n,s] = choices_savings[1][n,s] + states_a[1][n,s]*(1+paramsdec.r)
      states_b[2][n,s] = HC_prod(states_b[1][n,s], choices_x[1][n,s], shocks_b[n,s],
        paramsdec.iota0[1], paramsdec.iota1[1], paramsdec.iota2[1], paramsdec.iota3[1])
    end

  end

  return states_y, states_a, states_b, choices_savings, choices_x

end

# takes single argument which is composite of states, preferences, shocks, and parameters

function sim_choices(sim_choices_arg::SimChoiceArg)

   sim_choices(sim_choices_arg.initial_states, sim_choices_arg.hh_prefs, sim_choices_arg.shocks_y, sim_choices_arg.shocks_b, sim_choices_arg.paramsdec, sim_choices_arg.paramsshock)

end

#= Moment Generation =#

# calculate moments distribution of states and choices given S paths for each household

function moment_gen_dist(sim_choices; restrict_flag=1)

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
  savings_moments = Array{Array{Float64}}(Tminus1)
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
    savings_mom_t = zeros(2,S)
    x_mom_t = zeros(2,S)
    y_cov_t = zeros(2,S)
    a_cov_t = zeros(2,S)
    b_cov_t = zeros(2,S)

    for s in 1:S

      y_mom_t[:,s] = [mean(log.(sim_choices[1][t+1][:,s])) var(log.(sim_choices[1][t+1][:,s]))^0.5]
      a_mom_t[:,s] = [mean(log.(sim_choices[2][t+1][:,s])) var(log.(sim_choices[2][t+1][:,s]))^0.5]
      b_mom_t[:,s] = [mean(sim_choices[3][t+1][:,s]) var(sim_choices[3][t+1][:,s])^0.5]
      savings_mom_t[:,s] = [mean(sim_choices[4][t][:,s]) var(sim_choices[4][t][:,s])^0.5]
      x_mom_t[:,s] = [mean(sim_choices[5][t][:,s]) var(sim_choices[5][t][:,s])^0.5]
      y_cov_t[:,s] = [cov(log.(sim_choices[1][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[1][t][:,s]),sim_choices[5][t][:,s])]
      a_cov_t[:,s] = [cov(log.(sim_choices[2][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[2][t][:,s]),sim_choices[5][t][:,s])]
      b_cov_t[:,s] = [cov(log.(sim_choices[3][t][:,s]),sim_choices[4][t][:,s]) cov(log.(sim_choices[3][t][:,s]),sim_choices[5][t][:,s])]

    end

    # average moments over S paths
    y_moments[t] =  transpose(mean(y_mom_t,2))
    a_moments[t] = transpose(mean(a_mom_t,2))
    b_moments[t] = transpose(mean(b_mom_t,2))
    savings_moments[t] = transpose(mean(savings_mom_t,2))
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
      moment_stack  = vcat(moment_stack, savings_moments[t][i])
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

  return moment_stack, y_moments, a_moments, b_moments, savings_moments, x_moments, y_cov, a_cov, b_cov

end
