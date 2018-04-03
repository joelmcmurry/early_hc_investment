#= Utilities for Simulating from DGP =#

println("dgp utilities loading")

# include("/home/m/mcmurry2/Utilities_Solution_2P.jl")

include("Utilities_Solution_2P.jl")

#= Structures =#

mutable struct ParametersPrefs
  B_hi :: Float64 ## high future valuation
  B_lo :: Float64 ## low future valuation
  alphaT1_hi :: Float64 ##  high weight on assets
  alphaT1_lo :: Float64 ## low weight on assets

  gamma_y :: Array{Float64} ## type probability parameter vector for income (FIRST PARAMETER IS LOCKED DOWN AT 1)
  gamma_a :: Array{Float64} ## type probability parameter vector for assets
  gamma_b :: Array{Float64} ## type probability parameter vector for hc

  function ParametersPrefs(;B_hi=5., B_lo=1., alphaT1_hi=0.75, alphaT1_lo=0.25,
    gamma_y=[1., 1., 1., 1.], gamma_a=[1., 1., 1., 1.], gamma_b=[1., 1., 1., 1.])

    new(B_hi, B_lo, alphaT1_hi, alphaT1_lo, gamma_y, gamma_a, gamma_b)

  end

end

# create types given preference parameters

function type_construct(B_hi::Float64, B_lo::Float64, alphaT1_hi::Float64, alphaT1_lo::Float64)

  # construct vector of types from preferences
  type1 = [B_hi; alphaT1_hi]
  type2 = [B_hi; alphaT1_lo]
  type3 = [B_lo; alphaT1_hi]
  type4 = [B_lo; alphaT1_lo]

  type_vec = Array{Array}(4)
  type_vec[1] = type1
  type_vec[2] = type2
  type_vec[3] = type3
  type_vec[4] = type4

  return type_vec

end

mutable struct SimChoiceArg
   initial_states::Array{Float64}
   shocks_y::Array{Float64}
   shocks_b::Array{Float64}
   paramsprefs::ParametersPrefs
   paramsdec::ParametersDec
   paramsshock::ParametersShock
   error_log_flag::Int64

end

#= Type Probability Calculation =#

# compute function of linear combination of states

function type_prob_numerator(type_index::Int64, y0::Float64, a0::Float64, b0::Float64, paramsprefs::ParametersPrefs)

  computation_factor = 1.

  # logit = 1/(1+exp(-paramsprefs.gamma_y[type_index]*y0/computation_factor -
  #   paramsprefs.gamma_a[type_index]*a0/computation_factor - paramsprefs.gamma_b[type_index]*b0/computation_factor))

  numerator = paramsprefs.gamma_y[type_index]*y0/computation_factor +
      paramsprefs.gamma_a[type_index]*a0/computation_factor + paramsprefs.gamma_b[type_index]*b0/computation_factor

  return numerator

end

# rescale logist probabilities so type probabilities sum to 1

function type_prob(y0::Float64, a0::Float64, b0::Float64, paramsprefs::ParametersPrefs)

  numerator1 = type_prob_numerator(1, y0, a0, b0, paramsprefs)
  numerator2 = type_prob_numerator(2, y0, a0, b0, paramsprefs)
  numerator3 = type_prob_numerator(3, y0, a0, b0, paramsprefs)
  numerator4 = type_prob_numerator(4, y0, a0, b0, paramsprefs)

  denominator = numerator1 + numerator2 + numerator3 + numerator4

  prob1 = numerator1/denominator
  prob2 = numerator2/denominator
  prob3 = numerator3/denominator
  prob4 = numerator4/denominator

  return prob1, prob2, prob3, prob4

end

#= DGP Simulation =#

## For Each HH, Draw Preference Vector and Simulate S Paths of Shocks, Initial Conditions Given or Drawn Non-Parametrically from Data

function sim_paths(initial_state_data, paramsshock::ParametersShock;
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

  return initial_states, shocks_y, shocks_b

end

# divide paths up into N sections for parallel processing

function sim_paths_split(initial_states, shocks_y, shocks_b,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock; par_N=2, error_log_flag=0)

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
      shocks_y_section = shocks_y[index_start:index_end,:]
      shocks_b_section = shocks_b[index_start:index_end,:]

      # create and store simulated choice argument type with subset paths
      sim_choices_arg_section = SimChoiceArg(initial_states_section, shocks_y_section, shocks_b_section,
        paramsprefs, paramsdec, paramsshock, error_log_flag)
      sim_choices_arg_array[i] = sim_choices_arg_section

      # update indices
      index_start = index_end + 1
      index_end = min(index_start + subset_length, N)
   end

   return sim_choices_arg_array

end

## Given S Paths of Shocks, Solve HH Problem for Each Type and Store Average Choices and States Given Type Probabilities

# takes multiple arguments

function sim_choices(initial_states::Array{Float64}, shocks_y::Array{Float64}, shocks_b::Array{Float64},
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock; error_log_flag=0)

  # initialize error log
  error_log = Any[]

  # construct types given preferences
  type_vec = type_construct(paramsprefs.B_hi, paramsprefs.B_lo, paramsprefs.alphaT1_hi, paramsprefs.alphaT1_lo)

  # extract N, T, and S from initial states and shock paths
  N = length(initial_states[:,1])
  S = length(shocks_y[1,:])

  # store household state and decisions (will be averaged over types along S paths of shocks)
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

  # for each household, compute decisions for each type and compute average choice/state for each shock given type probability
  for n in 1:N

    # println(n)

    # calculate probability of each type given initial states
    type_probs_n = type_prob(states_y[1][n,1], states_a[1][n,1], states_b[1][n,1], paramsprefs)

    # initialize choices for endogenous starting values within HH (across types)
    choices0_type = zeros(2)

    for type_index in 1:4

      # store choices and states for HH and type
      choices_savings_n_type = zeros(S)
      choices_x_n_type = zeros(S)

      states_y_n_type = zeros(S)
      states_a_n_type = zeros(S)
      states_b_n_type = zeros(S)

      # start optimization with solution for previous type
      if type_index != 1
        aprime_start = choices0_type[1]
        x_start = choices0_type[2]
      else
        aprime_start = 1.
        x_start = 1.
      end

      # update preferences given type
      paramsdec.B = type_vec[type_index][1]
      paramsdec.alphaT1 = type_vec[type_index][2]
      paramsdec.alphaT2 = 1. - paramsdec.alphaT1

      # solve childhood period decisions
      V0_type, choices0_type, converged_check, iterations_taken, error_log_state = bellman_optim_child!(states_y[1][n,1],
        states_a[1][n,1], states_b[1][n,1], paramsdec, paramsshock,
        aprime_start=aprime_start, x_start=x_start, error_log_flag=error_log_flag)

      # record errors if prompted
      if error_log_flag == 1
        if isempty(error_log_state) == false
          push!(error_log, error_log_state)
        end
      end

      # store decisions conditional on type
      choices_savings_n_type = ones(S)*choices0_type[1] - ones(S)*states_a[1][n,1]*(1+paramsdec.r)
      choices_x_n_type = ones(S)*choices0_type[2]

      # calculate terminal period states given shocks conditional on type
      for s in 1:S
        states_y_n_type[s] = Y_evol(states_y[1][n,s], paramsdec.rho_y, shocks_y[n,s])
        states_a_n_type[s] = choices_savings_n_type[s] + states_a[1][n,s]*(1+paramsdec.r)
        states_b_n_type[s] = HC_prod(states_b[1][n,s], choices_x_n_type[s], shocks_b[n,s],
          paramsdec.iota0[1], paramsdec.iota1[1], paramsdec.iota2[1], paramsdec.iota3[1])
      end

      # weight choices and states by probability of type and add to type-averaged path along S shocks
      choices_savings[1][n,:] += choices_savings_n_type*type_probs_n[type_index]
      choices_x[1][n,:] += choices_x_n_type*type_probs_n[type_index]
      states_y[2][n,:] += states_y_n_type*type_probs_n[type_index]
      states_a[2][n,:] += states_a_n_type*type_probs_n[type_index]
      states_b[2][n,:] += states_b_n_type*type_probs_n[type_index]

    end

  end

  return states_y, states_a, states_b, choices_savings, choices_x, error_log

end

# takes single argument which is composite of states, preferences, shocks, and parameters

function sim_choices(sim_choices_arg::SimChoiceArg)

   sim_choices(sim_choices_arg.initial_states, sim_choices_arg.shocks_y, sim_choices_arg.shocks_b,
    sim_choices_arg.paramsprefs, sim_choices_arg.paramsdec, sim_choices_arg.paramsshock, error_log_flag=sim_choices_arg.error_log_flag)

end

#= Moment Generation =#

# calculate moments distribution of states and choices given S paths of shocks for each household

function moment_gen_dist(formatted_data; restrict_flag=1)

  # extract S given data
  S = length(formatted_data[1][1][1,:])

  # extract N given data
  N = length(formatted_data[1][1][:,1])

  ## Unconditional Moments

  # initialize storage of state and choice moments
  # y_moments = zeros(2)
  a_moments = zeros(2)
  b_moments = zeros(2)
  savings_moments = zeros(2)
  x_moments = zeros(2)
  y_cov = zeros(2)
  a_cov = zeros(2)
  b_cov = zeros(2)
  cov_s_x = zeros(1)

  # initialize s path specific moments

  # first and second (std dev) moments of terminal-period states and controls
  # y_mom_S = zeros(2,S)
  a_mom_S = zeros(2,S)
  b_mom_S = zeros(2,S)
  savings_mom_S = zeros(2,S)
  x_mom_S = zeros(2,S)

  # covariances of initial states and controls
  y_cov_S = zeros(2,S)
  a_cov_S = zeros(2,S)
  b_cov_S = zeros(2,S)

  # covariance of controls
  cov_s_x_S = zeros(1,S)

  # compute moments for each shock draw (and across draws from empirical distribution of initial conditions)
  for s in 1:S

    # y_mom_S[:,s] = [mean(log.(formatted_data[1][2][:,s])) var(log.(formatted_data[1][2][:,s]))^0.5]
    a_mom_S[:,s] = [mean(log.(formatted_data[2][2][:,s])) var(log.(formatted_data[2][2][:,s]))^0.5]
    b_mom_S[:,s] = [mean(formatted_data[3][2][:,s]) var(formatted_data[3][2][:,s])^0.5]
    savings_mom_S[:,s] = [mean(formatted_data[4][1][:,s]) var(formatted_data[4][1][:,s])^0.5]
    x_mom_S[:,s] = [mean(formatted_data[5][1][:,s]) var(formatted_data[5][1][:,s])^0.5]

    y_cov_S[:,s] = [cor(log.(formatted_data[1][1][:,s]),formatted_data[4][1][:,s]) cor(log.(formatted_data[1][1][:,s]),formatted_data[5][1][:,s])]
    a_cov_S[:,s] = [cor(log.(formatted_data[2][1][:,s]),formatted_data[4][1][:,s]) cor(log.(formatted_data[2][1][:,s]),formatted_data[5][1][:,s])]
    b_cov_S[:,s] = [cor(log.(formatted_data[3][1][:,s]),formatted_data[4][1][:,s]) cor(log.(formatted_data[3][1][:,s]),formatted_data[5][1][:,s])]

    cov_s_x_S[:,s] = cor(formatted_data[4][1][:,s],formatted_data[5][1][:,s])

  end

  # average moments over S paths
  # y_moments =  transpose(mean(y_mom_S,2))
  a_moments = transpose(mean(a_mom_S,2))
  b_moments = transpose(mean(b_mom_S,2))
  savings_moments = transpose(mean(savings_mom_S,2))
  x_moments = transpose(mean(x_mom_S,2))

  y_cov =  transpose(mean(y_cov_S,2))
  a_cov = transpose(mean(a_cov_S,2))
  b_cov = transpose(mean(b_cov_S,2))

  cov_s_x = transpose(mean(cov_s_x_S))

  # stack unconditional moments (means, std. devs, state/control covariances, control covariance)
  uncond_moment_stack = 0.
  for i in 1:2
    # uncond_moment_stack = vcat(uncond_moment_stack, y_moments[i])
    uncond_moment_stack = vcat(uncond_moment_stack, a_moments[i])
    uncond_moment_stack  = vcat(uncond_moment_stack, b_moments[i])
    uncond_moment_stack  = vcat(uncond_moment_stack, savings_moments[i])
    uncond_moment_stack  = vcat(uncond_moment_stack, x_moments[i])
  end

  uncond_moment_stack  = vcat(uncond_moment_stack, cov_s_x)

  for i in 1:2
    uncond_moment_stack = vcat(uncond_moment_stack, y_cov[i])
    uncond_moment_stack = vcat(uncond_moment_stack, a_cov[i])
    uncond_moment_stack  = vcat(uncond_moment_stack, b_cov[i])
  end

  # trim leading 0 that was necessary for initializtion before stack
  uncond_moment_stack = uncond_moment_stack[2:length(uncond_moment_stack)]

  ## Moments Conditional on Initial States (does not vary with shocks, so ignore S paths)

  # compute quantiles of initial states
  y_quantiles = quantile(formatted_data[1][1][:,1], [0.25,0.5,0.75])
  a_quantiles = quantile(formatted_data[2][1][:,1], [0.25,0.5,0.75])
  b_quantiles = quantile(formatted_data[3][1][:,1], [0.25,0.5,0.75])

  # initialize average conditional choices, conditioning on quantiles of initial state
  savings_cond_y = zeros(4)
  savings_cond_a = zeros(4)
  savings_cond_b = zeros(4)
  x_cond_y = zeros(4)
  x_cond_a = zeros(4)
  x_cond_b = zeros(4)

  # compute moments conditional on quantiles of initial income
  savings_cond_y[1] = mean(formatted_data[4][1][formatted_data[1][1][:,1].<y_quantiles[1],1])
  savings_cond_y[2] = mean(formatted_data[4][1][(formatted_data[1][1][:,1].>=y_quantiles[1])&(formatted_data[1][1][:,1].<y_quantiles[2]),1])
  savings_cond_y[3] = mean(formatted_data[4][1][(formatted_data[1][1][:,1].>=y_quantiles[2])&(formatted_data[1][1][:,1].<y_quantiles[3]),1])
  savings_cond_y[4] = mean(formatted_data[4][1][(formatted_data[1][1][:,1].>=y_quantiles[3]),1])

  x_cond_y[1] = mean(formatted_data[5][1][formatted_data[1][1][:,1].<y_quantiles[1],1])
  x_cond_y[2] = mean(formatted_data[5][1][(formatted_data[1][1][:,1].>=y_quantiles[1])&(formatted_data[1][1][:,1].<y_quantiles[2]),1])
  x_cond_y[3] = mean(formatted_data[5][1][(formatted_data[1][1][:,1].>=y_quantiles[2])&(formatted_data[1][1][:,1].<y_quantiles[3]),1])
  x_cond_y[4] = mean(formatted_data[5][1][(formatted_data[1][1][:,1].>=y_quantiles[3]),1])

  # compute moments conditional on quantiles of initial assets
  savings_cond_a[1] = mean(formatted_data[4][1][formatted_data[2][1][:,1].<a_quantiles[1],1])
  savings_cond_a[2] = mean(formatted_data[4][1][(formatted_data[2][1][:,1].>=a_quantiles[1])&(formatted_data[2][1][:,1].<a_quantiles[2]),1])
  savings_cond_a[3] = mean(formatted_data[4][1][(formatted_data[2][1][:,1].>=a_quantiles[2])&(formatted_data[2][1][:,1].<a_quantiles[3]),1])
  savings_cond_a[4] = mean(formatted_data[4][1][(formatted_data[2][1][:,1].>=a_quantiles[3]),1])

  x_cond_a[1] = mean(formatted_data[5][1][formatted_data[2][1][:,1].<a_quantiles[1],1])
  x_cond_a[2] = mean(formatted_data[5][1][(formatted_data[2][1][:,1].>=a_quantiles[1])&(formatted_data[2][1][:,1].<a_quantiles[2]),1])
  x_cond_a[3] = mean(formatted_data[5][1][(formatted_data[2][1][:,1].>=a_quantiles[2])&(formatted_data[2][1][:,1].<a_quantiles[3]),1])
  x_cond_a[4] = mean(formatted_data[5][1][(formatted_data[2][1][:,1].>=a_quantiles[3]),1])

  # compute moments conditional on quantiles of initial HC
  savings_cond_b[1] = mean(formatted_data[4][1][formatted_data[3][1][:,1].<b_quantiles[1],1])
  savings_cond_b[2] = mean(formatted_data[4][1][(formatted_data[3][1][:,1].>=b_quantiles[1])&(formatted_data[3][1][:,1].<b_quantiles[2]),1])
  savings_cond_b[3] = mean(formatted_data[4][1][(formatted_data[3][1][:,1].>=b_quantiles[2])&(formatted_data[3][1][:,1].<b_quantiles[3]),1])
  savings_cond_b[4] = mean(formatted_data[4][1][(formatted_data[3][1][:,1].>=b_quantiles[3]),1])

  x_cond_b[1] = mean(formatted_data[5][1][formatted_data[3][1][:,1].<b_quantiles[1],1])
  x_cond_b[2] = mean(formatted_data[5][1][(formatted_data[3][1][:,1].>=b_quantiles[1])&(formatted_data[3][1][:,1].<b_quantiles[2]),1])
  x_cond_b[3] = mean(formatted_data[5][1][(formatted_data[3][1][:,1].>=b_quantiles[2])&(formatted_data[3][1][:,1].<b_quantiles[3]),1])
  x_cond_b[4] = mean(formatted_data[5][1][(formatted_data[3][1][:,1].>=b_quantiles[3]),1])

  # stack conditional moments (by conditional quantile: savings conditional on quantile of each state, then investment conditional on quantile of each state)
  cond_moment_stack = 0
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, savings_cond_y[i])
  end
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, savings_cond_a[i])
  end
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, savings_cond_b[i])
  end
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, x_cond_y[i])
  end
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, x_cond_a[i])
  end
  for i in 1:4
    cond_moment_stack = vcat(cond_moment_stack, x_cond_b[i])
  end

  # trim leading 0 that was necessary for initializtion before stack
  cond_moment_stack = cond_moment_stack[2:length(cond_moment_stack)]

  ## Output

  # stack conditional and unconditional
  moments_out = [uncond_moment_stack; cond_moment_stack]

  # list of moments
  moments_desc = ["Mean lna2", "Mean b2", "Mean savings", "Mean x",
    "Stddev lna2", "Stddev b", "Stddev savings", "Stddev x",
    "Cor[s,x]",
    "Cor[savings,y]", "Cor[savings,a]", "Cor[savings,b]",
    "Cor[x,y]", "Cor[x,a]", "Cor[x,b]",
    "E[savings|1st quant. y]","E[savings|2nd quant. y]","E[savings|3rd quant. y]","E[savings|4th quant. y]",
    "E[savings|1st quant. a]","E[savings|2nd quant. a]","E[savings|3rd quant. a]","E[savings|4th quant. a]",
    "E[savings|1st quant. b]","E[savings|2nd quant. b]","E[savings|3rd quant. b]","E[savings|4th quant. b]",
    "E[x|1st quant. y]","E[x|2nd quant. y]","E[x|3rd quant. y]","E[x|4th quant. y]",
    "E[x|1st quant. a]","E[x|2nd quant. a]","E[x|3rd quant. a]","E[x|4th quant. a]",
    "E[x|1st quant. b]","E[x|2nd quant. b]","E[x|3rd quant. b]","E[x|4th quant. b]"]

  # moment restriction
  # moment_restrict = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  #   26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

  # restrict moments to "relevant" moments if prompted
  # if restrict_flag == 1
  #   moments_out = moments_out[moment_restrict]
  #   moments_desc = moments_desc[moment_restrict]
  # elseif restrict_flag == 0
  #   moments_out = moments_out
  #   moments_desc = moments_desc
  # else
  #   throw(error("restrict_flag must be 0 or 1"))
  # end

  return moments_out, moments_desc

end

#= Simulated Dataset =#

function sim_dataset(initial_states::Array{Float64})
