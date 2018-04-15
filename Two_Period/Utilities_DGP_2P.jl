#= Utilities for Simulating from DGP =#

println("dgp utilities loading")

# include("/home/m/mcmurry2/Utilities_Solution_2P.jl")

include("Utilities_Solution_2P.jl")

#= Structures =#

mutable struct ParametersPrefs
  sigma_alphaT1 :: Float64 ## common stddev of alphaT1
  sigma_alphaT2 :: Float64 ## common stddev of alphaT2
  rho :: Float64 ##  common correlation
  Sigma :: Array{Float64} ## var-covar matrix

  gamma_0 :: Array{Float64} ## intercept of mean function
  gamma_y :: Array{Float64} ## income coefficient of mean function
  gamma_a :: Array{Float64} ## asset coefficient of mean function
  gamma_b :: Array{Float64} ## hc coefficient of mean function

  function ParametersPrefs(;sigma_alphaT1=5., sigma_alphaT2=5., rho=0.75,
    gamma_0=[-10., 0.], gamma_y=[10., 5.], gamma_a=[-0.1, 5.], gamma_b=[5., 2.])

    Sigma = [sigma_alphaT1^2 rho*sigma_alphaT1*sigma_alphaT2; rho*sigma_alphaT1*sigma_alphaT2 sigma_alphaT2^2]

    new(sigma_alphaT1, sigma_alphaT2, rho, Sigma, gamma_0, gamma_y, gamma_a, gamma_b)

  end

end

mutable struct SimChoiceArg
   initial_states::Array{Float64}
   sample_prefs::Array{Float64}
   shocks_y::Array{Float64}
   shocks_b::Array{Float64}
   paramsprefs::ParametersPrefs
   paramsdec::ParametersDec
   paramsshock::ParametersShock
   error_log_flag::Int64
   bellman_trace::Bool
   bellman_iter::Int64
   bellman_tol::Float64

end

# create variance-covariance matrix of preferences

function pref_Sigma(sigma_alphaT1::Float64, sigma_alphaT2::Float64, rho::Float64)

  Sigma = Symmetric([sigma_alphaT1^2 rho*sigma_alphaT2*sigma_alphaT1; rho*sigma_alphaT1*sigma_alphaT2 sigma_alphaT2^2])

end

# draw state-specific types

function type_construct(y::Float64, a::Float64, b::Float64, paramsprefs::ParametersPrefs; seed=1234, type_N=2,
  alphaT1_lim_lb=0.00001, alphaT1_lim_ub=1000000., alphaT2_lim_lb=0.00001, alphaT2_lim_ub=1000000., mean_flag=0)

  # for computational reasons, set factor by which we divide states
  y_div = 1000000.
  a_div = 100000.
  b_div= 10.

  # compute mean of joint distribution given type
  mu_state = paramsprefs.gamma_0 + paramsprefs.gamma_y*y/y_div + paramsprefs.gamma_a*a/a_div + paramsprefs.gamma_b*b/b_div

  # draw N types or return mean draws only
  if mean_flag == 0
    srand(seed); type_vec = rand(MvNormal(mu_state, paramsprefs.Sigma), type_N)
  elseif mean_flag == 1
    type_vec = mu_state
  end

  # compute density of each draw
  type_pdf = pdf(MvNormal(mu_state, paramsprefs.Sigma), type_vec)

  # bound alphaT1 above 0
  type_vec[1,:] = exp.(type_vec[1,:])

  # bound alphaT2 above 0
  type_vec[2,:] = exp.(type_vec[2,:])

  # transform to probabilities
  type_prob = type_pdf./sum(type_pdf)

  # trim outliers
  type_vec[1,:] = max.(type_vec[1,:], alphaT1_lim_lb)
  type_vec[2,:] = max.(type_vec[2,:], alphaT2_lim_lb)

  type_vec[1,:] = min.(type_vec[1,:], alphaT1_lim_ub)
  type_vec[2,:] = min.(type_vec[2,:], alphaT2_lim_ub)

  return type_vec, type_prob

end

#= DGP Simulation =#

## Draw N Household/Type/Shock Observations

function sim_paths(initial_state_data, paramsshock::ParametersShock, paramsprefs::ParametersPrefs;
  seed=1234, N=1000, type_N=2)

  # draw initial states from EDF, advancing seed by 1
  srand(seed); initial_draws = rand(DiscreteUniform(1,length(initial_state_data[1][1])),N)
  seed += 1

  initial_states = [initial_state_data[1][1][initial_draws] initial_state_data[2][1][initial_draws] initial_state_data[3][1][initial_draws]]

  # extract unique initial states
  initial_states_unique = unique(initial_states,1)

  # initialize sample types
  sample_prefs = zeros(N,2)

  # random type draws, advancing seed by 1
  srand(seed); type_draws = rand(N)
  seed += 1

  # assign types to sample based on type probabilities, proceed by unique states
  for i in 1:length(initial_states_unique[:,1])
    # draw types and type probabilities
    type_vec, type_probs = type_construct(initial_states[i,1], initial_states[i,2], initial_states[i,3], paramsprefs, seed=seed, type_N=type_N)

    # find indices of draws with same initial states
    y_match = find(x->x==initial_states_unique[i,1],initial_states[:,1])
    a_match = find(x->x==initial_states_unique[i,2],initial_states[:,2])
    b_match = find(x->x==initial_states_unique[i,3],initial_states[:,3])
    init_state_indices = intersect(y_match, a_match, b_match)

    # assign types based on state-specific type probabilities (number of types is flexible)
    for j in 1:length(init_state_indices)
      if type_draws[init_state_indices[j]] <= type_probs[1]
        sample_prefs[init_state_indices[j],:] = type_vec[:,1]
      else
        for n in 2:type_N
          if type_draws[init_state_indices[j]] > sum(type_probs[1:n-1]) && type_draws[init_state_indices[j]] <= sum(type_probs[1:n])
            sample_prefs[init_state_indices[j],:] = type_vec[:,n]
          end
        end
      end
    end

  end

  # draw sample shocks
  srand(seed); shocks_y = rand(paramsshock.eps_y_dist,N)
  seed +=1
  srand(seed); shocks_b = rand(paramsshock.eps_b_dist,N)

  return initial_states, sample_prefs, shocks_y, shocks_b

end

# divide paths up into par_N sections for parallel processing

function sim_paths_split(initial_states, sample_prefs, shocks_y, shocks_b,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock; par_N=2, error_log_flag=0,
  bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

   # initialize array of simulated choice sections
   sim_choices_arg_array = Array{SimChoiceArg}(par_N)

   # extract number of unique initial conditions and  calculate subset length
   N = length(unique(initial_states,1)[:,1])
   subset_length = Int(ceil(N/par_N))

   # initialize indices for splitting
   index_start = 1
   index_end = subset_length

   for i in 1:par_N
      # subset unique initial conditions
      initial_states_unique_section = unique(initial_states,1)[index_start:index_end,:]

      # initialize sample and shock sections
      initial_states_section = zeros(1,3)
      sample_prefs_section = zeros(1,2)
      shocks_y_section = 0.
      shocks_b_section = 0.

      # loop through initial conditions and fill sample and shock sections with households of appropriate initial conditions
      for j in 1:length(initial_states_unique_section[:,1])

        # indentify all simulated individuals with these initial conditions
        y_match = find(x->x==initial_states_unique_section[j,1], initial_states[:,1])
        a_match = find(x->x==initial_states_unique_section[j,2], initial_states[:,2])
        b_match = find(x->x==initial_states_unique_section[j,3], initial_states[:,3])

        row_match = intersect(y_match, a_match, b_match)

        initial_states_section = vcat(initial_states_section, initial_states[row_match,:])
        sample_prefs_section = vcat(sample_prefs_section, sample_prefs[row_match,:])
        shocks_y_section = vcat(shocks_y_section, shocks_y[row_match,:])
        shocks_b_section = vcat(shocks_b_section, shocks_b[row_match,:])

      end

      # trim leading zeros used for initialization
      initial_states_section = initial_states_section[2:length(initial_states_section[:,1]),:]
      sample_prefs_section = sample_prefs_section[2:length(sample_prefs_section[:,1]),:]
      shocks_y_section = shocks_y_section[2:length(shocks_y_section[:,1]),:]
      shocks_b_section = shocks_b_section[2:length(shocks_b_section[:,1]),:]

      # create and store simulated choice argument type with subset paths
      sim_choices_arg_section = SimChoiceArg(initial_states_section, sample_prefs_section, shocks_y_section, shocks_b_section,
        paramsprefs, paramsdec, paramsshock, error_log_flag, bellman_trace, bellman_iter, bellman_tol)
      sim_choices_arg_array[i] = sim_choices_arg_section

      # update indices
      index_start = index_end + 1
      index_end = min(index_start + subset_length, N)

    end

   return sim_choices_arg_array

end

## Given Simulated Sample, Solve HH Problem

# takes multiple arguments

function sim_choices(initial_states::Array{Float64}, sample_prefs::Array{Float64}, shocks_y::Array{Float64}, shocks_b::Array{Float64},
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  error_log_flag=0, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  # initialize error log
  error_log = Any[]

  # extract N from sample
  N = length(initial_states[:,1])

  # extract unique initial conditions and types from sample
  initial_states_types_unique = unique(hcat(initial_states, sample_prefs),1)
  N_unique = length(initial_states_types_unique[:,1])

  # extract unique initial conditions from sample
  initial_states_unique = unique(initial_states,1)
  N_unique_states = length(initial_states_unique[:,1])

  # store household state and decisions
  states_y = Array{Array{Float64}}(2)
  states_a = Array{Array{Float64}}(2)
  states_b = Array{Array{Float64}}(2)
  choices_savings = Array{Array{Float64}}(1)
  choices_x = Array{Array{Float64}}(1)

  for t in 1:2
    # initialize states
    states_y[t] = zeros(N)
    states_a[t] = zeros(N)
    states_b[t] = zeros(N)

    # initialize choices
    if t == 1
      choices_savings[t] = zeros(N)
      choices_x[t] = zeros(N)
    end
  end

  # store initial state for each path and household
  states_y[1] = initial_states[:,1]
  states_a[1] = initial_states[:,2]
  states_b[1] = initial_states[:,3]

  # store decision rules by initial condition and type
  choices_lookup_init_type = zeros(N_unique,7)

  # initialize index for looping over both states and types
  init_type_index = 1

  # for each unique initial condition and type, compute choices and assign choices to sample
  for n in 1:N_unique_states

    # extract drawn types given initial conditions
    y_match = find(x->x==initial_states_unique[n,1], initial_states_types_unique[:,1])
    a_match = find(x->x==initial_states_unique[n,2], initial_states_types_unique[:,2])
    b_match = find(x->x==initial_states_unique[n,3], initial_states_types_unique[:,3])

    row_match = intersect(y_match, a_match, b_match)

    unique_prefs = unique(initial_states_types_unique[row_match,4:5],1)

    # initialize choices for endogenous starting values within HH (across types)
    choices0_type = zeros(2)

    for type_index in 1:length(unique_prefs[:,1])

      drawn_prefs = unique_prefs[type_index,:]

      # start optimization with solution for previous type
      # if type_index != 1
      #  aprime_start = choices0_type[1]
      #  x_start = choices0_type[2]
      # else
        aprime_start = 1.
        x_start = 1.
      # end

      # update preferences given type
      paramsdec.alphaT1 = drawn_prefs[1]
      paramsdec.alphaT2 = drawn_prefs[2]

      # println(initial_states_unique[n,1:3], drawn_prefs)

      # solve childhood period decisions
      V0_type, choices0_type, converged_check, iterations_taken, error_log_state = bellman_optim_child!(initial_states_unique[n,1],
        initial_states_unique[n,2], initial_states_unique[n,3], paramsdec, paramsshock,
        aprime_start=aprime_start, x_start=x_start, error_log_flag=error_log_flag,
        opt_trace=bellman_trace, opt_iter=bellman_iter, opt_tol=bellman_tol)

      # record errors if prompted
      if error_log_flag == 1
        if isempty(error_log_state) == false
          push!(error_log, error_log_state)
        end
      end

      # store decision rules by initial conditions and type
      choices_lookup_init_type[init_type_index,1:3] = initial_states_unique[n,1:3]
      choices_lookup_init_type[init_type_index,4:5] = drawn_prefs
      choices_lookup_init_type[init_type_index,6:7] = choices0_type

      # advance index
      init_type_index += 1

    end

  end

  # loop through sample and assign decision rules and compute state evolution
  for n in 1:N_unique

    # extract index of initial conditions and type
    y_match = find(x->x==choices_lookup_init_type[n,1], initial_states[:,1])
    a_match = find(x->x==choices_lookup_init_type[n,2], initial_states[:,2])
    b_match = find(x->x==choices_lookup_init_type[n,3], initial_states[:,3])
    B_match = find(x->x==choices_lookup_init_type[n,4], sample_prefs[:,1])
    alphaT1_match = find(x->x==choices_lookup_init_type[n,5], sample_prefs[:,2])

    row_match = intersect(y_match, a_match, b_match, alphaT1_match)

    # look up decisions given initial conditions and type
    choices_init_type = choices_lookup_init_type[n,6:7]

    # store decisions for this draw
    choices_savings[1][row_match] = choices_init_type[1] - states_a[1][row_match]*(1+paramsdec.r)
    choices_x[1][row_match] = choices_init_type[2]

    # calculate terminal period states given shocks
    for match_index in 1:length(row_match)
      states_y[2][row_match[match_index]] = Y_evol(states_y[1][row_match[match_index]], paramsdec.rho_y, shocks_y[row_match[match_index]])
      states_a[2][row_match[match_index]] = choices_savings[1][row_match[match_index]] + states_a[1][row_match[match_index]]*(1+paramsdec.r)
      states_b[2][row_match[match_index]] = HC_prod(states_b[1][row_match[match_index]], choices_x[1][row_match[match_index]], shocks_b[row_match[match_index]],
        paramsdec.iota0[1], paramsdec.iota1[1], paramsdec.iota2[1], paramsdec.iota3[1])
    end

  end

  return states_y, states_a, states_b, choices_savings, choices_x, sample_prefs, error_log

end

# takes single argument which is composite of states, preferences, shocks, and parameters

function sim_choices(sim_choices_arg::SimChoiceArg)

   sim_choices(sim_choices_arg.initial_states, sim_choices_arg.sample_prefs, sim_choices_arg.shocks_y, sim_choices_arg.shocks_b,
    sim_choices_arg.paramsprefs, sim_choices_arg.paramsdec, sim_choices_arg.paramsshock, error_log_flag=sim_choices_arg.error_log_flag,
    bellman_trace=sim_choices_arg.bellman_trace, bellman_iter=sim_choices_arg.bellman_iter, bellman_tol=sim_choices_arg.bellman_tol)

end

#= Moment Generation =#

# calculate moments distribution of states and choices given S paths of shocks for each household

function moment_gen_dist(formatted_data; restrict_flag=1)

  # extract NxM given data
  N = length(formatted_data[1][1][:,1])

  ## Unconditional Moments

  # initialize storage of state and choice moments
  # y_moments = zeros(2)
  a_moments = [mean(formatted_data[2][2]) var(formatted_data[2][2])^0.5]
  b_moments = [mean(formatted_data[3][2]) var(formatted_data[3][2])^0.5]
  savings_moments = [mean(formatted_data[4][1]) var(formatted_data[4][1])^0.5]
  x_moments = [mean(formatted_data[5][1]) var(formatted_data[5][1])^0.5]
  y_cov = [cor(formatted_data[1][1],formatted_data[4][1]) cor(formatted_data[1][1],formatted_data[5][1])]
  a_cov = [cor(formatted_data[2][1],formatted_data[4][1]) cor(formatted_data[2][1],formatted_data[5][1])]
  b_cov = [cor(formatted_data[3][1],formatted_data[4][1]) cor(formatted_data[3][1],formatted_data[5][1])]
  cov_s_x = cor(formatted_data[4][1],formatted_data[5][1])

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
  y_quantiles = quantile(formatted_data[1][1], [0.25,0.5,0.75])
  a_quantiles = quantile(formatted_data[2][1], [0.25,0.5,0.75])
  b_quantiles = quantile(formatted_data[3][1], [0.25,0.5,0.75])

  # initialize average conditional choices, conditioning on quantiles of initial state
  savings_cond_y = zeros(4)
  savings_cond_a = zeros(4)
  savings_cond_b = zeros(4)
  x_cond_y = zeros(4)
  x_cond_a = zeros(4)
  x_cond_b = zeros(4)

  # compute moments conditional on quantiles of initial income
  savings_cond_y[1] = mean(formatted_data[4][1][formatted_data[1][1].<y_quantiles[1],1])
  savings_cond_y[2] = mean(formatted_data[4][1][(formatted_data[1][1].>=y_quantiles[1])&(formatted_data[1][1].<y_quantiles[2]),1])
  savings_cond_y[3] = mean(formatted_data[4][1][(formatted_data[1][1].>=y_quantiles[2])&(formatted_data[1][1].<y_quantiles[3]),1])
  savings_cond_y[4] = mean(formatted_data[4][1][(formatted_data[1][1].>=y_quantiles[3]),1])

  x_cond_y[1] = mean(formatted_data[5][1][formatted_data[1][1].<y_quantiles[1],1])
  x_cond_y[2] = mean(formatted_data[5][1][(formatted_data[1][1].>=y_quantiles[1])&(formatted_data[1][1].<y_quantiles[2]),1])
  x_cond_y[3] = mean(formatted_data[5][1][(formatted_data[1][1].>=y_quantiles[2])&(formatted_data[1][1].<y_quantiles[3]),1])
  x_cond_y[4] = mean(formatted_data[5][1][(formatted_data[1][1].>=y_quantiles[3]),1])

  # compute moments conditional on quantiles of initial assets
  savings_cond_a[1] = mean(formatted_data[4][1][formatted_data[2][1].<a_quantiles[1],1])
  savings_cond_a[2] = mean(formatted_data[4][1][(formatted_data[2][1].>=a_quantiles[1])&(formatted_data[2][1].<a_quantiles[2]),1])
  savings_cond_a[3] = mean(formatted_data[4][1][(formatted_data[2][1].>=a_quantiles[2])&(formatted_data[2][1].<a_quantiles[3]),1])
  savings_cond_a[4] = mean(formatted_data[4][1][(formatted_data[2][1].>=a_quantiles[3]),1])

  x_cond_a[1] = mean(formatted_data[5][1][formatted_data[2][1].<a_quantiles[1],1])
  x_cond_a[2] = mean(formatted_data[5][1][(formatted_data[2][1].>=a_quantiles[1])&(formatted_data[2][1].<a_quantiles[2]),1])
  x_cond_a[3] = mean(formatted_data[5][1][(formatted_data[2][1].>=a_quantiles[2])&(formatted_data[2][1].<a_quantiles[3]),1])
  x_cond_a[4] = mean(formatted_data[5][1][(formatted_data[2][1].>=a_quantiles[3]),1])

  # compute moments conditional on quantiles of initial HC
  savings_cond_b[1] = mean(formatted_data[4][1][formatted_data[3][1].<b_quantiles[1],1])
  savings_cond_b[2] = mean(formatted_data[4][1][(formatted_data[3][1].>=b_quantiles[1])&(formatted_data[3][1].<b_quantiles[2]),1])
  savings_cond_b[3] = mean(formatted_data[4][1][(formatted_data[3][1].>=b_quantiles[2])&(formatted_data[3][1].<b_quantiles[3]),1])
  savings_cond_b[4] = mean(formatted_data[4][1][(formatted_data[3][1].>=b_quantiles[3]),1])

  x_cond_b[1] = mean(formatted_data[5][1][formatted_data[3][1].<b_quantiles[1],1])
  x_cond_b[2] = mean(formatted_data[5][1][(formatted_data[3][1].>=b_quantiles[1])&(formatted_data[3][1].<b_quantiles[2]),1])
  x_cond_b[3] = mean(formatted_data[5][1][(formatted_data[3][1].>=b_quantiles[2])&(formatted_data[3][1].<b_quantiles[3]),1])
  x_cond_b[4] = mean(formatted_data[5][1][(formatted_data[3][1].>=b_quantiles[3]),1])

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
  moments_desc = ["Mean a2", "Mean b2", "Mean savings", "Mean x",
    "Stddev a2", "Stddev b", "Stddev savings", "Stddev x",
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
  moment_omit = [3, 7]

  # restrict moments to "relevant" moments if prompted
  if restrict_flag == 1
    moments_out = setdiff(moments_out, moments_out[moment_omit])
    moments_desc = setdiff(moments_desc, moments_desc[moment_omit])
  elseif restrict_flag == 0
    moments_out = moments_out
    moments_desc = moments_desc
  else
    throw(error("restrict_flag must be 0 or 1"))
  end

  return moments_out, moments_desc

end
