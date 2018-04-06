#= Utilities for Estimation of Dynamic Chilhood Problem =#

using DataFrames
using Sobol
# using LatexPrint
using PyPlot

println("estimation utilities loading")

# include("/home/m/mcmurry2/Utilities_DGP_2P.jl")

include("Utilities_DGP_2P.jl")

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

      if ismissing(y_i_t) == false
         states_t[i,1] = y_i_t
      else
         states_t[i,1] = 1.
      end
      if ismissing(a_i_t) == false
         states_t[i,2] = a_i_t
      else
         states_t[i,2] = -1.
      end
      if ismissing(b_i_t) == false
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

#= DGP Simulation and Moment Generation =#

function dgp_moments(initial_state_data, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  seed=1234, N=1000, restrict_flag=1, error_log_flag=0, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  # simulate dataset
  sim_shocks = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=seed, N=N, type_N=type_N)

  sim_data = sim_choices(sim_shocks[1], sim_shocks[2], sim_shocks[3], sim_shocks[4],
    paramsprefs, paramsdec, paramsshock, error_log_flag=error_log_flag,
    bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

  # calculate simulated data moments
  sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

  return sim_moments, sim_data[1:6], sim_data[7]

end

# parallelize choices

function dgp_moments_par(initial_state_data, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  seed=1234, N=1000, restrict_flag=1, par_N=2, error_log_flag=0, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  # simulate dataset
  sim_shocks = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=seed, N=N, type_N=type_N)

  # split dataset and create objects that can be read by choice simulator
  split_sim_choice_arg = sim_paths_split(sim_shocks[1], sim_shocks[2], sim_shocks[3], sim_shocks[4],
   paramsprefs, paramsdec, paramsshock, par_N=par_N, error_log_flag=error_log_flag,
   bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

  # parallel compute choices
  sim_choices_par = pmap(sim_choices, split_sim_choice_arg)

  # stack split output
  states_y = Array{Array{Float64}}(2)
  states_a = Array{Array{Float64}}(2)
  states_b = Array{Array{Float64}}(2)
  choices_savings = Array{Array{Float64}}(1)
  choices_x = Array{Array{Float64}}(1)
  sample_prefs = []

  # initialize output with first parallel segment
  for t in 1:2
   states_y[t] = sim_choices_par[1][1][t]
   states_a[t] = sim_choices_par[1][2][t]
   states_b[t] = sim_choices_par[1][3][t]

   if t == 1
     choices_savings[t] = sim_choices_par[1][4][t]
     choices_x[t] = sim_choices_par[1][5][t]
   end
  end

  # stack sections
   for t in 1:2
      for par_segment in 2:par_N
         states_y[t] = vcat(states_y[t], sim_choices_par[par_segment][1][t])
         states_a[t] = vcat(states_a[t], sim_choices_par[par_segment][2][t])
         states_b[t] = vcat(states_b[t], sim_choices_par[par_segment][3][t])

         if t == 1
           choices_savings[t] = vcat(choices_savings[t], sim_choices_par[par_segment][4][t])
           choices_x[t] = vcat(choices_x[t], sim_choices_par[par_segment][5][t])
         end
      end
   end

   # stack drawn preferences
   for par_segment in 1:par_N
      sample_prefs = vcat(sample_prefs, sim_choices_par[par_segment][6])
   end

   sim_data = [states_y, states_a, states_b, choices_savings, choices_x, sample_prefs]

   # calculate simulated data moments
   sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

   # stack error logs
   error_log = Any[]
   if error_log_flag == 1
      for par_index in 1:par_N
         if isempty(sim_choices_par[par_index][6]) == false
            push!(error_log, sim_choices_par[par_index][6])
         end
      end
   end

  return sim_moments, sim_data, error_log

end

#= Estimation Utilities =#

## SMM Optimization Functions

# jointly estimate parameters via SMM

function smm(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  sigma_B_start=1., sigma_alphaT1_start=0.1, rho_start=0.,
  gamma_01_start=1., gamma_02_start=0.1, gamma_y1_start=0.1, gamma_y2_start=0.1,
  gamma_a1_start=0.1, gamma_a2_start=0.1, gamma_b1_start=0.1, gamma_b2_start=0.1,
  eps_b_var_start=0.022, iota0_start=1.87, iota1_start=0.42, iota2_start=0.06, iota3_start=0.0,
  N=1000,
  opt_code="neldermead", restrict_flag=1, seed=1234, error_log_flag=0,
  opt_trace=false, opt_iter=1000, print_flag=false, opt_tol=1e-9, par_flag=0, par_N=4,
  pref_only_flag=0, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments (if moment is nonzero, otherwise equal to 1)
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
     if data_moments[1][m] != 0.
       W[m,m] = 1/abs(data_moments[1][m])
    end
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  # define objective function given data moments
  smm_obj_inner(param_vec) = smm_obj(data_formatted, data_moments, W, param_vec,
    paramsprefs_float, paramsdec_float, paramsshock_float,
    N=N, restrict_flag=restrict_flag, seed=seed, error_log_flag=error_log_flag,
    print_flag=print_flag, par_flag=par_flag, par_N=par_N, pref_only_flag=pref_only_flag, type_N=type_N,
    bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

   if pref_only_flag == 0
      start_vec = [sigma_B_start, sigma_alphaT1_start, rho_start,
         gamma_01_start, gamma_02_start, gamma_y1_start, gamma_y2_start,
         gamma_a1_start, gamma_a2_start, gamma_b1_start, gamma_b2_start,
         eps_b_var_start, iota0_start, iota1_start, iota2_start, iota3_start]
   elseif pref_only_flag == 1
      start_vec = [sigma_B_start, sigma_alphaT1_start, rho_start,
         gamma_01_start, gamma_02_start, gamma_y1_start, gamma_y2_start,
         gamma_a1_start, gamma_a2_start, gamma_b1_start, gamma_b2_start]
   else
      throw(error("pref_only_flag must be 0 or 1"))
   end

  # minimize objective
  if opt_code == "neldermead"
    smm_opt = optimize(smm_obj_inner, start_vec, show_trace=opt_trace, iterations=opt_iter, g_tol=opt_tol)
  elseif opt_code == "lbfgs"
    smm_opt = optimize(smm_obj_inner, start_vec, LBFGS())
  elseif opt_code == "simulatedannealing"
    smm_opt = optimize(smm_obj_inner, start_vec, SimulatedAnnealing())
  else
    throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  return smm_opt

end

# solve SMM objective function on N Sobol points and save minimum

function smm_sobol(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   sobol_N=10,
   sigma_B_lb=0.001, sigma_B_ub=2., sigma_alphaT1_lb=0.001, sigma_alphaT1_ub=5.,
   rho_lb=-0.99, rho_ub=0.99,
   gamma_01_lb=-1., gamma_01_ub=2., gamma_02_lb=-1., gamma_02_ub=1.,
   gamma_y1_lb=-1., gamma_y1_ub=1., gamma_y2_lb=-1., gamma_y2_ub=1.,
   gamma_a1_lb=-1., gamma_a1_ub=1., gamma_a2_lb=-1., gamma_a2_ub=1.,
   gamma_b1_lb=-1., gamma_b1_ub=1., gamma_b2_lb=-1., gamma_b2_ub=1.,
   eps_b_var_lb=0.0001, eps_b_var_ub=1., iota0_lb=-2., iota0_ub=2., iota1_lb=0.0001, iota1_ub=2., iota2_lb=0.0001, iota2_ub=1., iota3_lb=-2., iota3_ub=2.,
   N=1000, restrict_flag=1, seed=1234, error_log_flag=0, print_flag=false,
   par_flag=0, par_N=4, pref_only_flag=0, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9,
   B_lim=1000., alphaT1_lim_lb=0.001, alphaT1_lim_ub=0.999)

   # store number of parameters
   if pref_only_flag == 0
      param_N = 16
   elseif pref_only_flag == 1
      param_N = 11
   end

   # initialize storage of parameter vectors and moments
   sobol_storage = Any[]

   # generate data moments
   data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

   # calculate weighting matrix with inverses of data moments
   W = eye(length(data_moments[1]))

   for m in 1:length(data_moments[1])
      if data_moments[1][m] != 0.
        W[m,m] = 1/abs(data_moments[1][m])
     end
   end

   # copy parameters structure to modify
   paramsprefs_float = deepcopy(paramsprefs)
   paramsdec_float = deepcopy(paramsdec)
   paramsshock_float = deepcopy(paramsshock)

   # construct Sobol sequence
   if pref_only_flag == 0
      s = SobolSeq(param_N, [sigma_B_lb, sigma_alphaT1_lb, rho_lb, gamma_01_lb, gamma_02_lb, gamma_y1_lb, gamma_y2_lb,
         gamma_a1_lb, gamma_a2_lb, gamma_b1_lb, gamma_b2_lb, eps_b_var_lb, iota0_lb, iota1_lb, iota2_lb, iota3_lb],
         [sigma_B_ub, sigma_alphaT1_ub, rho_ub, gamma_01_ub, gamma_02_ub, gamma_y1_ub, gamma_y2_ub,
         gamma_a1_ub, gamma_a2_ub, gamma_b1_ub, gamma_b2_ub,
         eps_b_var_ub, iota0_ub, iota1_ub, iota2_ub, iota3_ub])
   elseif pref_only_flag == 1
      s = SobolSeq(param_N, [sigma_B_lb, sigma_alphaT1_lb, rho_lb, gamma_01_lb, gamma_02_lb, gamma_y1_lb, gamma_y2_lb,
         gamma_a1_lb, gamma_a2_lb, gamma_b1_lb, gamma_b2_lb, eps_b_var_lb, iota0_lb, iota1_lb, iota2_lb, iota3_lb],
         [sigma_B_ub, sigma_alphaT1_ub, rho_ub, gamma_01_ub, gamma_02_ub, gamma_y1_ub, gamma_y2_ub,
         gamma_a1_ub, gamma_a2_ub, gamma_b1_ub, gamma_b2_ub])
   else
      throw(error("pref_only_flag must be 0 or 1"))
   end

   # skip initial portion of sequence per documentation (largest power of 2 less than sobol_N)
   skip(s, sobol_N)

   # initialize best guess
   min_obj_val = Inf
   min_obj_params = zeros(param_N)

   # loop through Sobol sequence and compute objective function (skip if constraints are violated or B or alphaT1 ordering is violated)
   for i in 1:sobol_N

      # initialize row of Sobol storage
      sobol_storage_i = zeros(param_N+39)

      param_sobol = next(s)

      sobol_storage_i[1:param_N] = param_sobol

      println(string("Iter ",i," of ",sobol_N,": ",param_sobol))

      # check is preference distribution is permissible
      median_expected_prefs = sobol_draw_check(data_formatted, [param_sobol[4], param_sobol[5]], [param_sobol[6], param_sobol[7]],
         [param_sobol[8], param_sobol[9]], [param_sobol[10], param_sobol[11]])

      if median_expected_prefs[1] > B_lim || median_expected_prefs[2] < alphaT1_lim_lb || median_expected_prefs[2] > alphaT1_lim_ub
         pref_error_flag = 1
      else
         pref_error_flag = 0
      end

      if param_sobol[1] <= 0. || param_sobol[2] <= 0. || param_sobol[3] < -1. || param_sobol[3] > 1. || (pref_only_flag == 0 && param_sobol[12] <= 0.) || pref_error_flag == 1
         obj_val = Inf
         println("skipped")

         sobol_storage_i[param_N+1:param_N+39] = fill(Inf,39)

      else
         obj_val, sim_moments, sim_data, error_log = smm_obj_moments(data_formatted, data_moments, W, param_sobol, paramsprefs_float, paramsdec_float, paramsshock_float,
            N=N, restrict_flag=restrict_flag, seed=seed, error_log_flag=error_log_flag, print_flag=print_flag,
            par_flag=par_flag, par_N=par_N, type_N=type_N, pref_only_flag=pref_only_flag,
            bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

         sobol_storage_i[param_N+1:param_N+39] = sim_moments[1]

      end

      if obj_val < min_obj_val
         min_obj_val = obj_val
         min_obj_params = param_sobol
      end

      # append moments and parameter vector to storage
      push!(sobol_storage, sobol_storage_i)

      # println(string("Min Value: ",min_obj_val," Minimizer: ",min_obj_params))

   end

  return min_obj_val, min_obj_params, sobol_storage

end

# check if mean preference draws for median household (in data) are permissible

function sobol_draw_check(data_formatted, gamma_0::Array{Float64}, gamma_y::Array{Float64}, gamma_a::Array{Float64}, gamma_b::Array{Float64})

   # for computational reasons, set factor by which we divide states
   y_div = 100000.
   a_div = 10000.
   b_div= 1.

   # calculate median state
   y_med = median(data_formatted[1][1])
   a_med = median(data_formatted[2][1])
   b_med = median(data_formatted[3][1])

   # calculate mean preferences for median state
   mu_state = gamma_0 + gamma_y*y_med/y_div + gamma_a*a_med/a_div + gamma_b*b_med/b_div

   # transform
   mean_prefs = [exp(mu_state[1]), 1/(1+exp(-mu_state[2]))]

   return mean_prefs

end

## SMM Optimization Write-To Functions

# runs function and writes output to text file in specified path

function smm_write_results(path, data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   sigma_B_start=1., sigma_alphaT1_start=0.1, rho_start=0.,
   gamma_01_start=1., gamma_02_start=0.1, gamma_y1_start=0.1, gamma_y2_start=0.1,
   gamma_a1_start=0.1, gamma_a2_start=0.1, gamma_b1_start=0.1, gamma_b2_start=0.1,
   eps_b_var_start=0.022, iota0_start=1.87, iota1_start=0.42, iota2_start=0.06, iota3_start=0.0,
   N=1000,
   opt_code="neldermead", restrict_flag=1, seed=1234, error_log_flag=0,
   opt_trace=false, opt_iter=1000, opt_tol=1e-9, print_flag=false, par_flag=0, par_N=4, type_N=2, pref_only_flag=0,
   bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

   # run SMM
   estimation_time = @elapsed estimation_result = smm(data_formatted, paramsprefs, paramsdec, paramsshock,
      sigma_B_start=sigma_B_start, sigma_alphaT1_start=sigma_alphaT1_start, rho_start=rho_start,
      gamma_01_start=gamma_01_start, gamma_02_start=gamma_02_start,
      gamma_y1_start=gamma_y1_start, gamma_y2_start=gamma_y2_start,
      gamma_a1_start=gamma_a1_start, gamma_a2_start=gamma_a2_start,
      gamma_b1_start=gamma_b1_start, gamma_b2_start=gamma_b2_start,
      eps_b_var_start=eps_b_var_start,
      iota0_start=iota0_start, iota1_start=iota1_start, iota2_start=iota2_start, iota3_start=iota3_start,
      N=N, opt_code=opt_code, restrict_flag=restrict_flag, seed=seed,
      error_log_flag=error_log_flag, opt_trace=opt_trace, opt_iter=opt_iter, opt_tol=opt_tol, print_flag=print_flag,
      par_flag=par_flag, par_N=par_N, type_N=type_N, bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

   # write to text file
   writedlm(path, transpose([estimation_time; estimation_result.minimum; estimation_result.minimizer]), ", ")

end

function smm_sobol_write_results(path_min, path_store, data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   sobol_N=10,
   sigma_B_lb=0.001, sigma_B_ub=2., sigma_alphaT1_lb=0.001, sigma_alphaT1_ub=5.,
   rho_lb=-0.99, rho_ub=0.99,
   gamma_01_lb=-1., gamma_01_ub=2., gamma_02_lb=-1., gamma_02_ub=1.,
   gamma_y1_lb=-1., gamma_y1_ub=1., gamma_y2_lb=-1., gamma_y2_ub=1.,
   gamma_a1_lb=-1., gamma_a1_ub=1., gamma_a2_lb=-1., gamma_a2_ub=1.,
   gamma_b1_lb=-1., gamma_b1_ub=1., gamma_b2_lb=-1., gamma_b2_ub=1.,
   eps_b_var_lb=0.0001, eps_b_var_ub=1., iota0_lb=-2., iota0_ub=2., iota1_lb=0.0001, iota1_ub=2., iota2_lb=0.0001, iota2_ub=1., iota3_lb=-2., iota3_ub=2.,
   N=1000, restrict_flag=1, seed=1234, error_log_flag=0, print_flag=false,
   par_flag=0, par_N=4, type_N=2, pref_only_flag=0, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9,
   B_lim=1000., alphaT1_lim_lb=0.001, alphaT1_lim_ub=0.999)

  # run SMM
  estimation_time = @elapsed estimation_result = smm_sobol(data_formatted, paramsprefs, paramsdec, paramsshock,
      sobol_N=sobol_N,
      sigma_B_lb=sigma_B_lb, sigma_alphaT1_lb=sigma_alphaT1_lb, rho_lb=rho_lb,
      gamma_01_lb=gamma_01_lb, gamma_02_lb=gamma_02_lb,
      gamma_y1_lb=gamma_y1_lb, gamma_y2_lb=gamma_y2_lb,
      gamma_a1_lb=gamma_a1_lb, gamma_a2_lb=gamma_a2_lb,
      gamma_b1_lb=gamma_b1_lb, gamma_b2_lb=gamma_b2_lb,
      eps_b_var_lb=eps_b_var_lb, iota0_lb=iota0_lb, iota1_lb=iota1_lb, iota2_lb=iota2_lb, iota3_lb=iota3_lb,
      sigma_B_ub=sigma_B_ub, sigma_alphaT1_ub=sigma_alphaT1_ub, rho_ub=rho_ub,
      gamma_01_ub=gamma_01_ub, gamma_02_ub=gamma_02_ub,
      gamma_y1_ub=gamma_y1_ub, gamma_y2_ub=gamma_y2_ub,
      gamma_a1_ub=gamma_a1_ub, gamma_a2_ub=gamma_a2_ub,
      gamma_b1_ub=gamma_b1_ub, gamma_b2_ub=gamma_b2_ub,
      eps_b_var_ub=eps_b_var_ub, iota0_ub=iota0_ub, iota1_ub=iota1_ub, iota2_ub=iota2_ub, iota3_ub=iota3_ub,
      N=N, restrict_flag=restrict_flag, seed=seed,
      error_log_flag=error_log_flag, print_flag=print_flag, par_flag=par_flag, par_N=par_N, type_N=type_N, pref_only_flag=pref_only_flag,
      bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol, B_lim=B_lim, alphaT1_lim_lb=alphaT1_lim_lb, alphaT1_lim_ub)

   # write minimizer to text file
   writedlm(path_min, transpose([estimation_time; estimation_result[1]; estimation_result[2]]), ", ")

   # write full list of moments and parameters in sequence to text file
   writecsv(path_store, estimation_result[3])

end

## Objecive Function for SMM Optimization, modifies parameters in place and takes target moments as argument

# jointly estimate all parameters, returning both objective function and simulated data/moments

function smm_obj_moments(initial_state_data, target_moments, W::Array, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, restrict_flag=1, seed=1234, error_log_flag=0,
  print_flag=false, par_flag=0, par_N=4, type_N=2, pref_only_flag=0, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  if print_flag == true
    println(param_vec)
  end

  # set parameters according to guess (and generate var-covar matrix)
  if pref_only_flag == 0
     paramsprefs.sigma_B, paramsprefs.sigma_alphaT1, paramsprefs.rho,
     paramsprefs.gamma_0[1], paramsprefs.gamma_0[2],
     paramsprefs.gamma_y[1], paramsprefs.gamma_y[2],
     paramsprefs.gamma_a[1], paramsprefs.gamma_a[2],
     paramsprefs.gamma_b[1], paramsprefs.gamma_b[2],
     paramsshock.eps_b_var, paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3 = param_vec

     paramsprefs.Sigma = Symmetric(pref_Sigma(param_vec[1], param_vec[2], param_vec[3]))
  elseif pref_only_flag == 1
     paramsprefs.sigma_B, paramsprefs.sigma_alphaT1, paramsprefs.rho,
     paramsprefs.gamma_0[1], paramsprefs.gamma_0[2],
     paramsprefs.gamma_y[1], paramsprefs.gamma_y[2],
     paramsprefs.gamma_a[1], paramsprefs.gamma_a[2],
     paramsprefs.gamma_b[1], paramsprefs.gamma_b[2] = param_vec

     paramsprefs.Sigma = Symmetric(pref_Sigma(param_vec[1], param_vec[2], param_vec[3]))
  end

  # relevant constraints
  if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] < -1. || param_vec[3] > 1. || (pref_only_flag == 0 && param_vec[12] <= 0.)
     obj = Inf
  else
    # simulate dataset and compute moments, whether serial or parallel
    if par_flag == 0
      sim_moments, sim_data, error_log = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock,
         seed=seed, N=N, restrict_flag=restrict_flag, error_log_flag=error_log_flag, type_N=type_N,
         bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)
   elseif par_flag == 1
      sim_moments, sim_data, error_log = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock,
        seed=seed, N=N, restrict_flag=restrict_flag, par_N=par_N, error_log_flag=error_log_flag, type_N=type_N,
        bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)
   else
      throw(error("par_flag must be 0 or 1"))
   end

    # calculate SMM objective with weighting matrix
    obj = (transpose(sim_moments[1] - target_moments[1])*W*(sim_moments[1] - target_moments[1]))[1]

  end

  if print_flag == true
    println(obj)
  end

  return obj, sim_moments, sim_data, error_log

end

# return only objective function

function smm_obj(initial_state_data, target_moments, W::Array, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, restrict_flag=1, seed=1234, error_log_flag=0,
  print_flag=false, par_flag=0, par_N=4, type_N=2, pref_only_flag=0,
  bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

  obj = smm_obj_moments(initial_state_data, target_moments, W, param_vec, paramsprefs, paramsdec, paramsshock,
   N=N, restrict_flag=restrict_flag, seed=seed, error_log_flag=error_log_flag, print_flag=print_flag, par_flag=par_flag, par_N=par_N,
   type_N=type_N, pref_only_flag=pref_only_flag, bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)[1]

   return obj

end

## Objecive Function for SMM, does not modify parameters in place

# jointly estimate all parameters

function smm_obj_testing(data_formatted, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, restrict_flag=1, seed=1234, error_log_flag=0,
  par_flag=0, par_N=4, type_N=2, pref_only_flag=0,
  bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9, print_flag=false)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments (if moment is not zero)
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
     if data_moments[1][m] != 0.
       W[m,m] = 1/abs(data_moments[1][m])
    end
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  println(param_vec)

  # compute objective function
  obj, sim_moments, sim_data, error_log = smm_obj_moments(data_formatted, data_moments, W, param_vec,
    paramsprefs_float, paramsdec_float, paramsshock_float, N=N, restrict_flag=restrict_flag, seed=restrict_flag, error_log_flag=error_log_flag,
    print_flag=print_flag, par_flag=par_flag, par_N=par_N, type_N=type_N, pref_only_flag=pref_only_flag,
    bellman_trace=bellman_trace, bellman_iter=bellman_iter, bellman_tol=bellman_tol)

  # find moment that results in maximum error (weighted by W entry)
  if isnan(maximum(((sim_moments[1] - data_moments[1]).^2).*diag(W)))==false
    max_error_index = find(x->x==maximum(((sim_moments[1] - data_moments[1]).^2).*diag(W)),((sim_moments[1] - data_moments[1]).^2).*diag(W))
  else
    max_error_index = NaN
  end

  return obj, max_error_index[1], data_moments, sim_moments, sim_data, error_log

end

#= Informative Moment Graphing =#

# plot function

function plot_series(param_grid, moment_series, param_name, moment_name)

  plot_fig = figure()
  plot(param_grid, moment_series)
  xlabel(param_name)
  ylabel(moment_name)
  ax = PyPlot.gca()
  ax[:legend](loc="lower right")
  title(string(moment_name, " Varying ", param_name))

end

# plot moments varying single parameter

function plot_moments(param_grid, stored_moments, param_name; moment_display_flag=1)

  N = length(param_grid)

  # initialize moment arrays
  a2_mean = zeros(N)
  b2_mean = zeros(N)
  s_mean = zeros(N)
  x_mean = zeros(N)
  a2_sd = zeros(N)
  b2_sd = zeros(N)
  s_sd = zeros(N)
  x_sd = zeros(N)
  cor_s_x = zeros(N)

  cor_y1_s = zeros(N)
  cor_a1_s = zeros(N)
  cor_b1_s = zeros(N)
  cor_y1_x = zeros(N)
  cor_a1_x = zeros(N)
  cor_b1_x = zeros(N)

  e_s_y1 = zeros(N)
  e_s_y2 = zeros(N)
  e_s_y3 = zeros(N)
  e_s_y4 = zeros(N)
  e_s_a1 = zeros(N)
  e_s_a2 = zeros(N)
  e_s_a3 = zeros(N)
  e_s_a4 = zeros(N)
  e_s_b1 = zeros(N)
  e_s_b2 = zeros(N)
  e_s_b3 = zeros(N)
  e_s_b4 = zeros(N)

  e_x_y1 = zeros(N)
  e_x_y2 = zeros(N)
  e_x_y3 = zeros(N)
  e_x_y4 = zeros(N)
  e_x_a1 = zeros(N)
  e_x_a2 = zeros(N)
  e_x_a3 = zeros(N)
  e_x_a4 = zeros(N)
  e_x_b1 = zeros(N)
  e_x_b2 = zeros(N)
  e_x_b3 = zeros(N)
  e_x_b4 = zeros(N)

  # fill moment arrays
  for n in 1:N
     a2_mean[n] = stored_moments[n][1]
     b2_mean[n] = stored_moments[n][2]
     s_mean[n] = stored_moments[n][3]
     x_mean[n] = stored_moments[n][4]
     a2_sd[n] = stored_moments[n][5]
     b2_sd[n] = stored_moments[n][6]
     s_sd[n] = stored_moments[n][7]
     x_sd[n] = stored_moments[n][8]
     cor_s_x[n] = stored_moments[n][9]

     cor_y1_s[n] = stored_moments[n][10]
     cor_a1_s[n] = stored_moments[n][11]
     cor_b1_s[n] = stored_moments[n][12]
     cor_y1_x[n] = stored_moments[n][13]
     cor_a1_x[n] = stored_moments[n][14]
     cor_b1_x[n] = stored_moments[n][15]

     e_s_y1[n] = stored_moments[n][16]
     e_s_y2[n] = stored_moments[n][17]
     e_s_y3[n] = stored_moments[n][18]
     e_s_y4[n] = stored_moments[n][19]
     e_s_a1[n] = stored_moments[n][20]
     e_s_a2[n] = stored_moments[n][21]
     e_s_a3[n] = stored_moments[n][22]
     e_s_a4[n] = stored_moments[n][23]
     e_s_b1[n] = stored_moments[n][24]
     e_s_b2[n] = stored_moments[n][25]
     e_s_b3[n] = stored_moments[n][26]
     e_s_b4[n] = stored_moments[n][27]

     e_x_y1[n] = stored_moments[n][28]
     e_x_y2[n] = stored_moments[n][29]
     e_x_y3[n] = stored_moments[n][30]
     e_x_y4[n] = stored_moments[n][31]
     e_x_a1[n] = stored_moments[n][32]
     e_x_a2[n] = stored_moments[n][33]
     e_x_a3[n] = stored_moments[n][34]
     e_x_a4[n] = stored_moments[n][35]
     e_x_b1[n] = stored_moments[n][36]
     e_x_b2[n] = stored_moments[n][37]
     e_x_b3[n] = stored_moments[n][38]
     e_x_b4[n] = stored_moments[n][39]
  end

  ## Plot Selected Series

  if moment_display_flag == 1
     # mean graphs
     plot_series(param_grid, a2_mean, param_name, "Mean a2")
     plot_series(param_grid, b2_mean, param_name, "Mean b2")
     plot_series(param_grid, s_mean, param_name, "Mean savings")
     plot_series(param_grid, x_mean, param_name, "Mean investment")

     # std dev graphs
     plot_series(param_grid, a2_sd, param_name, "Std. Dev. a2")
     plot_series(param_grid, b2_sd, param_name, "Std. Dev. b2")
     plot_series(param_grid, s_sd, param_name, "Std. Dev. savings")
     plot_series(param_grid, x_sd, param_name, "Std. Dev. investment")

     # control corr
     plot_series(param_grid, cor_s_x, param_name, "Cor[savings, investment]")
  end


  if moment_display_flag == 2
     # corr of state and control
     plot_series(param_grid, cor_y1_s, param_name, "Cor(savings, y1)")
     plot_series(param_grid, cor_a1_s, param_name, "Cor(savings, a1)")
     plot_series(param_grid, cor_b1_s, param_name, "Cor(savings, b1)")
     plot_series(param_grid, cor_y1_x, param_name, "Cor(investment, y1)")
     plot_series(param_grid, cor_a1_x, param_name, "Cor(investment, a1)")
     plot_series(param_grid, cor_b1_x, param_name, "Cor(investment, b1)")
  end

  if moment_display_flag == 3
     # conditional averages of savings
     plot_series(param_grid, e_s_y1, param_name, "E[savings | 1st y quantile]")
     plot_series(param_grid, e_s_y2, param_name, "E[savings | 2nd y quantile]")
     plot_series(param_grid, e_s_y3, param_name, "E[savings | 3rd y quantile]")
     plot_series(param_grid, e_s_y4, param_name, "E[savings | 4th y quantile]")
     plot_series(param_grid, e_s_a1, param_name, "E[savings | 1st a quantile]")
     plot_series(param_grid, e_s_a2, param_name, "E[savings | 2nd a quantile]")
     plot_series(param_grid, e_s_a3, param_name, "E[savings | 3rd a quantile]")
     plot_series(param_grid, e_s_a4, param_name, "E[savings | 4th a quantile]")
     plot_series(param_grid, e_s_b1, param_name, "E[savings | 1st b quantile]")
     plot_series(param_grid, e_s_b2, param_name, "E[savings | 2nd b quantile]")
     plot_series(param_grid, e_s_b3, param_name, "E[savings | 3rd b quantile]")
     plot_series(param_grid, e_s_b4, param_name, "E[savings | 4th b quantile]")
  end

  if moment_display_flag == 4
     # conditional averages of investment
     plot_series(param_grid, e_x_y1, param_name, "E[investment | 1st y quantile]")
     plot_series(param_grid, e_x_y2, param_name, "E[investment | 2nd y quantile]")
     plot_series(param_grid, e_x_y3, param_name, "E[investment | 3rd y quantile]")
     plot_series(param_grid, e_x_y4, param_name, "E[investment | 4th y quantile]")
     plot_series(param_grid, e_x_a1, param_name, "E[investment | 1st a quantile]")
     plot_series(param_grid, e_x_a2, param_name, "E[investment | 2nd a quantile]")
     plot_series(param_grid, e_x_a3, param_name, "E[investment | 3rd a quantile]")
     plot_series(param_grid, e_x_a4, param_name, "E[investment | 4th a quantile]")
     plot_series(param_grid, e_x_b1, param_name, "E[investment | 1st b quantile]")
     plot_series(param_grid, e_x_b2, param_name, "E[investment | 2nd b quantile]")
     plot_series(param_grid, e_x_b3, param_name, "E[investment | 3rd b quantile]")
     plot_series(param_grid, e_x_b4, param_name, "E[investment | 4th b quantile]")
  end

end

## Parameter Specific Code

function vary_param(param_name::String, initial_state_data, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
  param_lower::Float64, param_upper::Float64, param_N::Int64;
  N=1000, restrict_flag=1, seed=1234, par_flag=0, par_N=2, error_log_flag=0, type_N=2, pref_only_flag=0)

  # create parameter grid
  param_grid = linspace(param_lower, param_upper, param_N)

  # create object to store moments at each parameter value
  moment_storage = Array{Array{Float64}}(param_N)

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  # set parameters according to guess
  if pref_only_flag == 0
     paramsprefs_float.sigma_B, paramsprefs_float.sigma_alphaT1, paramsprefs_float.rho,
     paramsprefs_float.gamma_0[1], paramsprefs_float.gamma_0[2],
     paramsprefs_float.gamma_a[1], paramsprefs_float.gamma_a[2],
     paramsprefs_float.gamma_b[1], paramsprefs_float.gamma_b[2],
     paramsshock_float.eps_b_var, paramsdec_float.iota0, paramsdec_float.iota1, paramsdec_float.iota2, paramsdec_float.iota3 = param_vec
  elseif pref_only_flag == 1
     paramsprefs_float.sigma_B, paramsprefs_float.sigma_alphaT1, paramsprefs_float.rho,
     paramsprefs_float.gamma_0[1], paramsprefs_float.gamma_0[2],
     paramsprefs_float.gamma_a[1], paramsprefs_float.gamma_a[2],
     paramsprefs_float.gamma_b[1], paramsprefs_float.gamma_b[2] = param_vec
  end

  # for each parameter guess, solve model and calculate full set of moments and store
  for n in 1:param_N
   println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
   if param_name == "sigma_B"
      paramsprefs_float.sigma_B = param_grid[n]
   elseif param_name == "sigma_alphaT1"
      paramsprefs_float.sigma_alphaT1 = param_grid[n]
   elseif param_name == "rho"
      paramsprefs_float.rho = param_grid[n]
   elseif param_name == "eps_b_var"
      paramsshock_float.eps_b_var = param_grid[n]
   elseif param_name == "iota0"
      paramsdec_float.iota0 = param_grid[n]
   elseif param_name == "iota1"
      paramsdec_float.iota1 = param_grid[n]
   elseif param_name == "iota2"
      paramsdec_float.iota2 = param_grid[n]
   elseif param_name == "iota3"
      paramsdec_float.iota3 = param_grid[n]
   elseif param_name == "gamma_01"
      paramsprefs_float.gamma_0[1] = param_grid[n]
   elseif param_name == "gamma_02"
      paramsprefs_float.gamma_0[2] = param_grid[n]
   elseif param_name == "gamma_y1"
      paramsprefs_float.gamma_y[1] = param_grid[n]
   elseif param_name == "gamma_y2"
      paramsprefs_float.gamma_y[2] = param_grid[n]
   elseif param_name == "gamma_a1"
      paramsprefs_float.gamma_a[1] = param_grid[n]
   elseif param_name == "gamma_a2"
      paramsprefs_float.gamma_a[2] = param_grid[n]
   elseif param_name == "gamma_b1"
      paramsprefs_float.gamma_b[1] = param_grid[n]
   elseif param_name == "gamma_b2"
      paramsprefs_float.gamma_b[2] = param_grid[n]
   else
      throw(error("invalid parameter name"))
   end

    if param_vec[1] <= 0. || param_vec[2] <= 0. || param_vec[3] < -1. || param_vec[3] > 1. || (pref_only_flag == 0 && param_vec[10] <= 0.)
      obj = Inf
    else
      # simulate dataset and compute moments, whether serial or parallel
      if par_flag == 0
         sim_moments, sim_data, error_log = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
            seed=seed, N=N, restrict_flag=restrict_flag, error_log_flag=error_log_flag, type_N=type_N, pref_only_flag=pref_only_flag)
      elseif par_flag == 1
         sim_moments, sim_data, error_log = dgp_moments_par(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
            seed=seed, N=N, restrict_flag=restrict_flag, par_N=par_N, error_log_flag=error_log_flag, type_N=type_N, pref_only_flag=pref_only_flag)
      else
         throw(error("par_flag must be 0 or 1"))
      end

      # store moments
      moment_storage[n] = sim_moments[1]

    end

  end

  return param_grid, moment_storage, param_name

end

#= Informative Moment Graphing - Quantiles =#

# plot quantiles of a moment, varying bin in which single parameter is "fixed"

function plot_moment_quantiles(moment_index::Int64, param_index::Int64, param_quantiles, moment_storage_25, moment_storage_50, moment_storage_75, data_moments)

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

   param_desc = ["sigma_B", "sigma_alphaT1", "rho", "gamma_01", "gamma_02",
      "gamma_y1","gamma_y2", "gamma_a1","gamma_a2", "gamma_b1","gamma_b2","eps_b_var","iota0","iota1","iota2","iota3"]

   moment_name = moments_desc[moment_index]

   param_name = param_desc[param_index]

   plot_fig = figure()
   plot(param_quantiles, moment_storage_25[:,moment_index],label="25th Perc.")
   plot(param_quantiles, moment_storage_50[:,moment_index],label="50th Perc.")
   plot(param_quantiles, moment_storage_75[:,moment_index],label="75th Perc.")
   plot(param_quantiles, ones(length(param_quantiles))*data_moments[moment_index],label="data")
   xlabel(param_name)
   ylabel(moment_name)
   ax = PyPlot.gca()
   ax[:legend](loc="lower right")
   title(string(moment_name, " Varying ", param_name))

end

# generate plots for all moments and one parameter

function plot_moment_quantiles_all_moments(param_index::Int64, param_quantiles, moment_storage_25, moment_storage_50, moment_storage_75, data_moments)

   for i in 1:length(moment_storage_25[1,:])
      plot_moment_quantiles(i, param_index, param_quantiles, moment_storage_25, moment_storage_50, moment_storage_75, data_moments)
   end

end

# generate plots for all parameters and one moment

function plot_moment_quantiles_all_params(moment_index::Int64, data_moments, path_storage; bin_N=5)

   for i in 1:20
      compute_quantile_output = param_constant_quantile(i, path_storage)

      plot_moment_quantiles(moment_index, i, compute_quantile_output[2], compute_quantile_output[3], compute_quantile_output[4], compute_quantile_output[5], data_moments)
   end

end

# compute quantiles holding single parameter constant (within a bin)

function param_constant_quantile(param_index::Int, path_storage; bin_N=5)

   # import stored sobol sequence of parameters and moments
   sobol_storage = readcsv(path_storage)

   # extract number of moments
   N_moments = length(sobol_storage[1,21:length(sobol_storage[1,:])])

   # subset stored sequence to parameter vectors that did not violate constraints
   sobol_storage_valid = sobol_storage[find(x->x!=Inf, sobol_storage[:,21]),:]

   # compute bins for paramter to hold "fixed"
   param_quantiles = quantile(sobol_storage_valid[:,param_index], (linspace(1,bin_N,bin_N)-ones(bin_N))/(bin_N-1))

   ## For each bin, compute 25th, 50th, and 75th quantiles of each moment

   # initialize moment storage
   moment_storage_25 = zeros(bin_N,N_moments)
   moment_storage_50 = zeros(bin_N,N_moments)
   moment_storage_75 = zeros(bin_N,N_moments)

   for n in 1:bin_N
      # find indices of fixed parameter in proper bin
      if n == 1
         bin_indices = find(x->x<=param_quantiles[n], sobol_storage_valid[:,param_index])
      else
         bin_indices = intersect(find(x->x>param_quantiles[n-1], sobol_storage_valid[:,param_index]),
            find(x->x<=param_quantiles[n], sobol_storage_valid[:,param_index]))
      end

      for moment_index in 1:N_moments
         moment_storage_25[n,moment_index] = quantile(sobol_storage_valid[bin_indices, 20+moment_index], 0.25)
         moment_storage_50[n,moment_index] = quantile(sobol_storage_valid[bin_indices, 20+moment_index], 0.5)
         moment_storage_75[n,moment_index] = quantile(sobol_storage_valid[bin_indices, 20+moment_index], 0.75)
      end

   end

   return param_index, param_quantiles, moment_storage_25, moment_storage_50, moment_storage_75

end
