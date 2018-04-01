#= Utilities for Estimation of Dynamic Chilhood Problem =#

using DataFrames
using Sobol
# using LatexPrint
# using PyPlot

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
  T=2, seed=1234, S=10, N=100, sample_code="nodraw", restrict_flag=1)

  # simulate dataset
  sim_shocks = sim_paths(initial_state_data, paramsshock,
          T=T, seed=seed, N=N, S=S, sample_code=sample_code)

  sim_data = sim_choices(sim_shocks[1], sim_shocks[2][1], sim_shocks[3][1],
    paramsprefs, paramsdec, paramsshock)

  # calculate simulated data moments
  sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

  return sim_moments

end

# parallelize choices

function dgp_moments_par(initial_state_data, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  T=2, seed=1234, S=10, N=100, sample_code="nodraw", restrict_flag=1, par_N=2)

  # simulate dataset
  sim_shocks = sim_paths(initial_state_data, paramsshock,
          T=T, seed=seed, N=N, S=S, sample_code=sample_code)

  # split dataset and create objects that can be read by choice simulator
  split_sim_choice_arg = sim_paths_split(sim_shocks[1], sim_shocks[2][1], sim_shocks[3][1], paramsprefs, paramsdec, paramsshock, par_N=par_N)

  # parallel compute choices
  sim_choices_par = pmap(sim_choices, split_sim_choice_arg)

  # stack split output
  states_y = Array{Array{Float64}}(2)
  states_a = Array{Array{Float64}}(2)
  states_b = Array{Array{Float64}}(2)
  choices_savings = Array{Array{Float64}}(1)
  choices_x = Array{Array{Float64}}(1)

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

   sim_data = [states_y, states_a, states_b, choices_savings, choices_x]

   # calculate simulated data moments
   sim_moments = moment_gen_dist(sim_data, restrict_flag=restrict_flag)

  return sim_moments

end

#= Estimation Utilities =#

## SMM Optimization Functions

# jointly estimate parameters via SMM

function smm(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  B_hi_start=5., B_lo_start=1., alphaT1_hi_start=0.75, alphaT1_mid_start=0.5, alphaT1_lo_start=0.25,
  gamma_y1_start=1., gamma_y2_start=1., gamma_y3_start=1., gamma_y4_start=1., gamma_y5_start=1., gamma_y6_start=1.,
  gamma_a1_start=1., gamma_a2_start=1., gamma_a3_start=1., gamma_a4_start=1., gamma_a5_start=1., gamma_a6_start=1.,
  gamma_b1_start=1., gamma_b2_start=1., gamma_b3_start=1., gamma_b4_start=1., gamma_b5_start=1., gamma_b6_start=1.,
  eps_b_var_start=0.022, iota0_start=1.87, iota1_start=0.42, iota2_start=0.06, iota3_start=0.0,
  N=100, T=2, S=100,
  opt_code="neldermead", sample_code="nodraw", restrict_flag=1, seed=1234, error_log_flag=0,
  opt_trace=false, opt_iter=1000, print_flag=true, opt_tol=1e-9, par_flag=0, par_N=4)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments (if moment is nonzero, otherwise equal to 1)
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
     if data_moments[1][m] != 0.
       W[m,m] = 1/abs(data_moments[1][m])^2
    end
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  # define objective function given data moments
  smm_obj_inner(param_vec) = smm_obj(data_formatted, data_moments, W, param_vec,
    paramsprefs_float, paramsdec_float, paramsshock_float,
    N=N, T=T, restrict_flag=restrict_flag, S=S, seed=seed, sample_code=sample_code, error_log_flag=error_log_flag,
    print_flag=print_flag, par_flag=par_flag, par_N=par_N)

  # minimize objective
  if opt_code == "neldermead"
    smm_opt = optimize(smm_obj_inner, [B_hi_start, B_lo_start, alphaT1_hi_start, alphaT1_mid_start, alphaT1_lo_start,
       gamma_y1_start, gamma_y2_start, gamma_y3_start, gamma_y4_start, gamma_y5_start, gamma_y6_start,
       gamma_a1_start, gamma_a2_start, gamma_a3_start, gamma_a4_start, gamma_a5_start, gamma_a6_start,
       gamma_b1_start, gamma_b2_start, gamma_b3_start, gamma_b4_start, gamma_b5_start, gamma_b6_start,
       eps_b_var_start, iota0_start, iota1_start, iota2_start, iota3_start], show_trace=opt_trace, iterations=opt_iter, g_tol=opt_tol)
  elseif opt_code == "lbfgs"
    smm_opt = optimize(smm_obj_inner, [B_hi_start, B_lo_start, alphaT1_hi_start, alphaT1_mid_start, alphaT1_lo_start,
       gamma_y1_start, gamma_y2_start, gamma_y3_start, gamma_y4_start, gamma_y5_start, gamma_y6_start,
       gamma_a1_start, gamma_a2_start, gamma_a3_start, gamma_a4_start, gamma_a5_start, gamma_a6_start,
       gamma_b1_start, gamma_b2_start, gamma_b3_start, gamma_b4_start, gamma_b5_start, gamma_b6_start,
       eps_b_var_start, iota0_start, iota1_start, iota2_start, iota3_start], LBFGS())
  elseif opt_code == "simulatedannealing"
    smm_opt = optimize(smm_obj_inner, [B_hi_start, B_lo_start, alphaT1_hi_start, alphaT1_mid_start, alphaT1_lo_start,
       gamma_y1_start, gamma_y2_start, gamma_y3_start, gamma_y4_start, gamma_y5_start, gamma_y6_start,
       gamma_a1_start, gamma_a2_start, gamma_a3_start, gamma_a4_start, gamma_a5_start, gamma_a6_start,
       gamma_b1_start, gamma_b2_start, gamma_b3_start, gamma_b4_start, gamma_b5_start, gamma_b6_start,
       eps_b_var_start, iota0_start, iota1_start, iota2_start, iota3_start], SimulatedAnnealing())
  else
    throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  return smm_opt

end

# solve SMM objective function on N Sobol points and save minimum

function smm_sobol(data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   sobol_N=10,
   B_hi_lb=1., B_hi_ub=10., B_lo_lb=1., B_lo_ub=5.,
   alphaT1_hi_lb=0.5, alphaT1_hi_ub=0.99, alphaT1_mid_lb=0.25, alphaT1_mid_ub=0.75, alphaT1_lo_lb=0.01, alphaT1_lo_ub=0.25,
   gamma_y1_lb=1., gamma_y1_ub=2., gamma_y2_lb=1., gamma_y2_ub=2., gamma_y3_lb=1., gamma_y3_ub=2., gamma_y4_lb=1., gamma_y4_ub=2., gamma_y5_lb=1., gamma_y5_ub=2., gamma_y6_lb=1., gamma_y6_ub=2.,
   gamma_a1_lb=1., gamma_a1_ub=2., gamma_a2_lb=1., gamma_a2_ub=2., gamma_a3_lb=1., gamma_a3_ub=2., gamma_a4_lb=1., gamma_a4_ub=2., gamma_a5_lb=1., gamma_a5_ub=2., gamma_a6_lb=1., gamma_a6_ub=2.,
   gamma_b1_lb=1., gamma_b1_ub=2., gamma_b2_lb=1., gamma_b2_ub=2., gamma_b3_lb=1., gamma_b3_ub=2., gamma_b4_lb=1., gamma_b4_ub=2., gamma_b5_lb=1., gamma_b5_ub=2., gamma_b6_lb=1., gamma_b6_ub=2.,
   eps_b_var_lb=0.0001, eps_b_var_ub=0.1, iota0_lb=-2., iota0_ub=2., iota1_lb=0.0001, iota1_ub=2., iota2_lb=0.0001, iota2_ub=1., iota3_lb=-2., iota3_ub=2.,
   N=100, T=2, S=100, sample_code="nodraw", restrict_flag=1, seed=1234, error_log_flag=0, print_flag=false,
   par_flag=0, par_N=4)

   # generate data moments
   data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

   # calculate weighting matrix with inverses of data moments
   W = eye(length(data_moments[1]))

   for m in 1:length(data_moments[1])
      if data_moments[1][m] != 0.
        W[m,m] = 1/abs(data_moments[1][m])^2
     end
   end

   # copy parameters structure to modify
   paramsprefs_float = deepcopy(paramsprefs)
   paramsdec_float = deepcopy(paramsdec)
   paramsshock_float = deepcopy(paramsshock)

   # construct Sobol sequence
   s = SobolSeq(28, [B_hi_lb, B_lo_lb, alphaT1_hi_lb, alphaT1_mid_lb, alphaT1_lo_lb, gamma_y1_lb, gamma_y2_lb, gamma_y3_lb, gamma_y4_lb, gamma_y5_lb, gamma_y6_lb,
      gamma_a1_lb, gamma_a2_lb, gamma_a3_lb, gamma_a4_lb, gamma_a5_lb, gamma_a6_lb, gamma_b1_lb, gamma_b2_lb, gamma_b3_lb, gamma_b4_lb,  gamma_b5_lb, gamma_b6_lb,
      eps_b_var_lb, iota0_lb, iota1_lb, iota2_lb, iota3_lb], [B_hi_ub, B_lo_ub, alphaT1_hi_ub, alphaT1_mid_ub, alphaT1_lo_ub, gamma_y1_ub, gamma_y2_ub, gamma_y3_ub, gamma_y4_ub,
      gamma_y5_ub, gamma_y6_ub, gamma_a1_ub, gamma_a2_ub, gamma_a3_ub, gamma_a4_ub, gamma_a5_ub, gamma_a6_ub, gamma_b1_ub, gamma_b2_ub, gamma_b3_ub, gamma_b4_ub,  gamma_b5_ub, gamma_b6_ub,
      eps_b_var_ub, iota0_ub, iota1_ub, iota2_ub, iota3_ub])

   # skip initial portion of sequence per documentation (largest power of 2 less than sobol_N)
   skip(s, sobol_N)

   # initialize best guess
   min_obj_val = Inf
   min_obj_params = zeros(28)

   # loop through Sobol sequence and compute objective function (skip if constraints are violated or B or alphaT1 ordering is violated)
   for i in 1:sobol_N

      param_sobol = next(s)

      println(string("Iter ",i," of ",sobol_N,": ",param_sobol))

      if param_sobol[1] < 1. || param_sobol[2] < 1. || param_sobol[3] <= 0. || param_sobol[3] >= 1. ||
         param_sobol[4] <= 0. || param_sobol[4] >= 1. || param_sobol[5] <= 0. || param_sobol[5] >= 1. ||
         param_sobol[24] <= 0. || param_sobol[3] < param_sobol[4] || param_sobol[3] < param_sobol[5] ||
         param_sobol[4] < param_sobol[5] || param_sobol[1] < param_sobol[2]
         obj_val = Inf
      else
         obj_val = smm_obj(data_formatted, data_moments, W, param_sobol, paramsprefs_float, paramsdec_float, paramsshock_float,
            N=N, T=T, restrict_flag=restrict_flag, S=S, seed=seed, sample_code=sample_code, error_log_flag=error_log_flag, print_flag=print_flag,
            par_flag=par_flag, par_N=par_N)
      end

      if obj_val < min_obj_val
         min_obj_val = obj_val
         min_obj_params = param_sobol
      end

      # println(string("Min Value: ",min_obj_val," Minimizer: ",min_obj_params))

   end

  return min_obj_val, min_obj_params

end

## SMM Optimization Write-To Functions

# runs function and writes output to text file in specified path

function smm_write_results(path, data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   B_hi_start=5., B_lo_start=1., alphaT1_hi_start=0.75, alphaT1_mid_start=0.5, alphaT1_lo_start=0.25,
   gamma_y1_start=1., gamma_y2_start=1., gamma_y3_start=1., gamma_y4_start=1., gamma_y5_start=1., gamma_y6_start=1.,
   gamma_a1_start=1., gamma_a2_start=1., gamma_a3_start=1., gamma_a4_start=1., gamma_a5_start=1., gamma_a6_start=1.,
   gamma_b1_start=1., gamma_b2_start=1., gamma_b3_start=1., gamma_b4_start=1., gamma_b5_start=1., gamma_b6_start=1.,
   eps_b_var_start=0.022, iota0_start=1.87, iota1_start=0.42, iota2_start=0.06, iota3_start=0.0,
  N=100, T=2, S=100,
  opt_code="neldermead", sample_code="nodraw", restrict_flag=1, seed=1234, error_log_flag=0,
  opt_trace=false, opt_iter=1000, opt_tol=1e-9, print_flag=true, par_flag=0, par_N=4)

   # run SMM
   estimation_time = @elapsed estimation_result = smm(data_formatted, paramsprefs, paramsdec, paramsshock,
      B_hi_start=B_hi_start, B_lo_start=B_lo_start, alphaT1_hi_start=alphaT1_hi_start, alphaT1_mid_start=alphaT1_mid_start, alphaT1_lo_start=alphaT1_lo_start,
      gamma_y1_start=gamma_y1_start, gamma_y2_start=gamma_y2_start, gamma_y3_start=gamma_y3_start, gamma_y4_start=gamma_y4_start, gamma_y5_start=gamma_y5_start, gamma_y6_start=gamma_y6_start,
      gamma_a1_start=gamma_a1_start, gamma_a2_start=gamma_a2_start, gamma_a3_start=gamma_a3_start, gamma_a4_start=gamma_a4_start, gamma_a5_start=gamma_a5_start, gamma_a6_start=gamma_a6_start,
      gamma_b1_start=gamma_b1_start, gamma_b2_start=gamma_b2_start, gamma_b3_start=gamma_b3_start, gamma_b4_start=gamma_b4_start, gamma_b5_start=gamma_b5_start, gamma_b6_start=gamma_b6_start,
      eps_b_var_start=eps_b_var_start,
      iota0_start=iota0_start, iota1_start=iota1_start, iota2_start=iota2_start, iota3_start=iota3_start,
      N=N, T=T, S=S, opt_code=opt_code, sample_code=sample_code, restrict_flag=restrict_flag, seed=seed,
      error_log_flag=error_log_flag, opt_trace=opt_trace, opt_iter=opt_iter, opt_tol=opt_tol, print_flag=print_flag,
      par_flag=par_flag, par_N=par_N)

   # write to text file
   writedlm(path, transpose([estimation_time; estimation_result.minimum; estimation_result.minimizer]), ", ")

end

function smm_sobol_write_results(path, data_formatted, paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
   sobol_N=10, B_hi_lb=1., B_hi_ub=10., B_lo_lb=1., B_lo_ub=5.,
   alphaT1_hi_lb=0.5, alphaT1_hi_ub=0.99, alphaT1_mid_lb=0.25, alphaT1_mid_ub=0.75, alphaT1_lo_lb=0.01, alphaT1_lo_ub=0.25,
   gamma_y1_lb=1., gamma_y1_ub=2., gamma_y2_lb=1., gamma_y2_ub=2., gamma_y3_lb=1., gamma_y3_ub=2., gamma_y4_lb=1., gamma_y4_ub=2., gamma_y5_lb=1., gamma_y5_ub=2., gamma_y6_lb=1., gamma_y6_ub=2.,
   gamma_a1_lb=1., gamma_a1_ub=2., gamma_a2_lb=1., gamma_a2_ub=2., gamma_a3_lb=1., gamma_a3_ub=2., gamma_a4_lb=1., gamma_a4_ub=2., gamma_a5_lb=1., gamma_a5_ub=2., gamma_a6_lb=1., gamma_a6_ub=2.,
   gamma_b1_lb=1., gamma_b1_ub=2., gamma_b2_lb=1., gamma_b2_ub=2., gamma_b3_lb=1., gamma_b3_ub=2., gamma_b4_lb=1., gamma_b4_ub=2., gamma_b5_lb=1., gamma_b5_ub=2., gamma_b6_lb=1., gamma_b6_ub=2.,
   eps_b_var_lb=0.0001, eps_b_var_ub=0.1, iota0_lb=-2., iota0_ub=2., iota1_lb=0.0001, iota1_ub=2., iota2_lb=0.0001, iota2_ub=1., iota3_lb=-2., iota3_ub=2.,
   N=100, T=2, S=100, sample_code="nodraw", restrict_flag=1, seed=1234, error_log_flag=0, print_flag=false,
   par_flag=0, par_N=4)

  # run SMM
  estimation_time = @elapsed estimation_result = smm_sobol(data_formatted, paramsprefs, paramsdec, paramsshock,
      sobol_N=sobol_N,
      B_hi_lb=B_hi_lb, B_lo_lb=B_lo_lb, alphaT1_hi_lb=alphaT1_hi_lb, alphaT1_mid_lb=alphaT1_mid_lb, alphaT1_lo_lb=alphaT1_lo_lb,
      gamma_y1_lb=gamma_y1_lb, gamma_y2_lb=gamma_y2_lb, gamma_y3_lb=gamma_y3_lb, gamma_y4_lb=gamma_y4_lb, gamma_y5_lb=gamma_y5_lb, gamma_y6_lb=gamma_y6_lb,
      gamma_a1_lb=gamma_a1_lb, gamma_a2_lb=gamma_a2_lb, gamma_a3_lb=gamma_a3_lb, gamma_a4_lb=gamma_a4_lb, gamma_a5_lb=gamma_a5_lb, gamma_a6_lb=gamma_a6_lb,
      gamma_b1_lb=gamma_b1_lb, gamma_b2_lb=gamma_b2_lb, gamma_b3_lb=gamma_b2_lb, gamma_b4_lb=gamma_b2_lb, gamma_b5_lb=gamma_b5_lb, gamma_b6_lb=gamma_b6_lb,
      eps_b_var_lb=eps_b_var_lb, iota0_lb=iota0_lb, iota1_lb=iota1_lb, iota2_lb=iota1_lb, iota3_lb=iota3_lb,
      B_hi_ub=B_hi_ub, B_lo_ub=B_lo_ub, alphaT1_hi_ub=alphaT1_hi_ub, alphaT1_mid_ub=alphaT1_mid_ub, alphaT1_lo_ub=alphaT1_lo_ub,
      gamma_y1_ub=gamma_y1_ub, gamma_y2_ub=gamma_y2_ub, gamma_y3_ub=gamma_y3_ub, gamma_y4_ub=gamma_y4_ub, gamma_y5_ub=gamma_y5_ub, gamma_y6_ub=gamma_y6_ub,
      gamma_a1_ub=gamma_a1_ub, gamma_a2_ub=gamma_a2_ub, gamma_a3_ub=gamma_a3_ub, gamma_a4_ub=gamma_a4_ub, gamma_a5_ub=gamma_a5_ub, gamma_a6_ub=gamma_a5_ub,
      gamma_b1_ub=gamma_b1_ub, gamma_b2_ub=gamma_b2_ub, gamma_b3_ub=gamma_b3_ub, gamma_b4_ub=gamma_b4_ub, gamma_b5_ub=gamma_b5_ub, gamma_b6_ub=gamma_b6_ub,
      eps_b_var_ub=eps_b_var_ub, iota0_ub=iota0_ub, iota1_ub=iota1_ub, iota2_ub=iota2_ub, iota3_ub=iota3_ub,
      N=N, T=T, S=S, sample_code=sample_code, restrict_flag=restrict_flag, seed=seed,
      error_log_flag=error_log_flag, print_flag=print_flag, par_flag=par_flag, par_N=par_N)

   # write to text file
   writedlm(path, transpose([estimation_time; estimation_result[1]; estimation_result[2]]), ", ")

end

## Objecive Function for SMM Optimization, modifies parameters in place and takes target moments as argument

# jointly estimate all parameters

function smm_obj(initial_state_data, target_moments, W::Array, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw", error_log_flag=0,
  print_flag=false, par_flag=0, par_N=4)

  if print_flag == true
    println(param_vec)
  end

  # set parameters according to guess
  paramsprefs.B_hi, paramsprefs.B_lo, paramsprefs.alphaT1_hi, paramsprefs.alphaT1_mid, paramsprefs.alphaT1_lo,
  paramsprefs.gamma_y[1], paramsprefs.gamma_y[2], paramsprefs.gamma_y[3], paramsprefs.gamma_y[4], paramsprefs.gamma_y[5], paramsprefs.gamma_y[6],
  paramsprefs.gamma_a[1], paramsprefs.gamma_a[2], paramsprefs.gamma_a[3], paramsprefs.gamma_a[4], paramsprefs.gamma_a[5], paramsprefs.gamma_a[6],
  paramsprefs.gamma_b[1], paramsprefs.gamma_b[2], paramsprefs.gamma_b[3], paramsprefs.gamma_b[4], paramsprefs.gamma_b[5], paramsprefs.gamma_b[6],
  paramsshock.eps_b_var, paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3 = param_vec

  # relevant constraints
  if param_vec[1] < 1. || param_vec[2] < 1. || param_vec[3] <= 0. || param_vec[3] >= 1. ||
     param_vec[4] <= 0. || param_vec[4] >= 1. || param_vec[5] <= 0. || param_vec[5] >= 1. ||
     param_vec[24] <= 0. || param_vec[3] < param_vec[4] || param_vec[3] < param_vec[5] ||
     param_vec[4] < param_vec[5] || param_vec[1] < param_vec[2]
     obj = Inf
  else
    # simulate dataset and compute moments, whether serial or parallel
    if par_flag == 0
       sim_moments = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock,
         T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=restrict_flag)
   elseif par_flag == 1
      sim_moments = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock,
        T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=restrict_flag, par_N=par_N)
   else
      throw(error("par_flag must be 0 or 1"))
   end

    # calculate SMM objective with weighting matrix
    obj = (transpose(sim_moments[1] - target_moments[1])*W*(sim_moments[1] - target_moments[1]))[1]

  end

  if print_flag == true
    println(obj)
  end

  return obj

end

## Objecive Function for SMM, does not modify parameters in place

# jointly estimate all parameters

function smm_obj_testing(data_formatted, param_vec::Array,
  paramsprefs::ParametersPrefs, paramsdec::ParametersDec, paramsshock::ParametersShock;
  N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw", error_log_flag=0,
  par_flag=0, par_N=4)

  # generate data moments
  data_moments = moment_gen_dist(data_formatted, restrict_flag=restrict_flag)

  # calculate weighting matrix with inverses of data moments (if moment is not zero)
  W = eye(length(data_moments[1]))

  for m in 1:length(data_moments[1])
     if data_moments[1][m] != 0.
       W[m,m] = 1/abs(data_moments[1][m])^2
    end
  end

  # copy parameters structure to modify
  paramsprefs_float = deepcopy(paramsprefs)
  paramsdec_float = deepcopy(paramsdec)
  paramsshock_float = deepcopy(paramsshock)

  println(param_vec)

  # set parameters according to guess
  paramsprefs_float.B_hi, paramsprefs_float.B_lo, paramsprefs_float.alphaT1_hi, paramsprefs_float.alphaT1_mid, paramsprefs_float.alphaT1_lo,
  paramsprefs_float.gamma_y[1], paramsprefs_float.gamma_y[2], paramsprefs_float.gamma_y[3], paramsprefs_float.gamma_y[4], paramsprefs_float.gamma_y[5], paramsprefs_float.gamma_y[6],
  paramsprefs_float.gamma_a[1], paramsprefs_float.gamma_a[2], paramsprefs_float.gamma_a[3], paramsprefs_float.gamma_a[4], paramsprefs_float.gamma_a[5], paramsprefs_float.gamma_a[6],
  paramsprefs_float.gamma_b[1], paramsprefs_float.gamma_b[2], paramsprefs_float.gamma_b[3], paramsprefs_float.gamma_b[4], paramsprefs_float.gamma_b[5], paramsprefs_float.gamma_b[6],
  paramsshock_float.eps_b_var, paramsdec_float.iota0, paramsdec_float.iota1, paramsdec_float.iota2, paramsdec_float.iota3 = param_vec

  # relevant constraints
  if param_vec[1] < 1. || param_vec[2] < 1. || param_vec[3] <= 0. || param_vec[3] >= 1. ||
     param_vec[4] <= 0. || param_vec[4] >= 1. || param_vec[5] <= 0. || param_vec[5] >= 1. ||
     param_vec[24] <= 0. || param_vec[3] < param_vec[4] || param_vec[3] < param_vec[5] ||
     param_vec[4] < param_vec[5] || param_vec[1] < param_vec[2]
     obj = Inf
  else
     # simulate dataset and compute moments, whether serial or parallel
      if par_flag == 0
         sim_moments = dgp_moments(data_formatted, paramsprefs_float, paramsdec_float, paramsshock_float,
         T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=restrict_flag)
      elseif par_flag == 1
         sim_moments = dgp_moments_par(data_formatted, paramsprefs_float, paramsdec_float, paramsshock,
         T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=restrict_flag, par_N=par_N)
      else
         throw(error("par_flag must be 0 or 1"))
      end

    # calculate SMM objective with identity weighting matrix
    obj = (transpose(sim_moments[1] - data_moments[1])*W*(sim_moments[1] - data_moments[1]))[1]

  end

  # find moment that results in maximum error (weighted by W entry)
  if isnan(maximum(((sim_moments[1] - data_moments[1]).^2).*diag(W)))==false
    max_error_index = find(x->x==maximum(((sim_moments[1] - data_moments[1]).^2).*diag(W)),((sim_moments[1] - data_moments[1]).^2).*diag(W))
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

  return obj, max_error_index[1], data_moments, sim_moments

end

#= Informative Moment Graphing =#

# plot function
#
# function plot_series(param_grid, moment_series, param_name, moment_name)
#
#   plot_fig = figure()
#   plot(param_grid, moment_series)
#   xlabel(param_name)
#   ylabel(moment_name)
#   ax = PyPlot.gca()
#   ax[:legend](loc="lower right")
#   title(string(moment_name, " Varying ", param_name))
#
# end
#
#
# # plot moments varying single parameter
#
# function plot_moments(param_grid, stored_moments, param_name)
#
#   N = length(param_grid)
#
#   # initialize moment arrays
#   lna2_mean = zeros(N)
#   b2_mean = zeros(N)
#   s_mean = zeros(N)
#   x_mean = zeros(N)
#   lna2_sd = zeros(N)
#   b2_sd = zeros(N)
#   s_sd = zeros(N)
#   x_sd = zeros(N)
#   cov_y1_s = zeros(N)
#   cov_y1_x = zeros(N)
#   cov_a1_s = zeros(N)
#   cov_a1_x = zeros(N)
#   cov_b1_s = zeros(N)
#   cov_b1_x = zeros(N)
#
#   # fill moment arrays
#   for n in 1:N
#     lna2_mean[n] = stored_moments[n][1]
#     b2_mean[n] = stored_moments[n][2]
#     s_mean[n] = stored_moments[n][3]
#     x_mean[n] = stored_moments[n][4]
#     lna2_sd[n] = stored_moments[n][5]
#     b2_sd[n] = stored_moments[n][6]
#     s_sd[n] = stored_moments[n][7]
#     x_sd[n] = stored_moments[n][8]
#     cov_y1_s[n] = stored_moments[n][9]
#     cov_y1_x[n] = stored_moments[n][10]
#     cov_a1_s[n] = stored_moments[n][11]
#     cov_a1_x[n] = stored_moments[n][12]
#     cov_b1_s[n] = stored_moments[n][13]
#     cov_b1_x[n] = stored_moments[n][14]
#   end
#
#   # mean graphs
#   plot_series(param_grid, lna2_mean, param_name, "Mean log(a2)")
#   plot_series(param_grid, b2_mean, param_name, "Mean b2")
#   plot_series(param_grid, s_mean, param_name, "Mean savings")
#   plot_series(param_grid, x_mean, param_name, "Mean investment")
#
#   # std dev graphs
#   plot_series(param_grid, lna2_sd, param_name, "Std. Dev. log(a2)")
#   plot_series(param_grid, b2_sd, param_name, "Std. Dev. b2")
#   plot_series(param_grid, s_sd, param_name, "Std. Dev. savings")
#   plot_series(param_grid, x_sd, param_name, "Std. Dev. investment")
#
#   # cov with s graphs
#   plot_series(param_grid, cov_y1_s, param_name, "Cov(log(y1),savings)")
#   plot_series(param_grid, cov_a1_s, param_name, "Cov(log(a1),savings)")
#   plot_series(param_grid, cov_b1_s, param_name, "Cov(b1,savings)")
#
#   # cov with x graphs
#   plot_series(param_grid, cov_y1_x, param_name, "Cov(log(y1),investment)")
#   plot_series(param_grid, cov_a1_x, param_name, "Cov(log(a1),investment)")
#   plot_series(param_grid, cov_b1_x, param_name, "Cov(b1,investment)")
#
# end
#
#
# function vary_mu1(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsprefs_float.mu[1] = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "mu1")
#
# end
#
# function vary_mu2(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsprefs_float.mu[2] = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "mu2")
#
# end
#
# function vary_sigma1(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
#
#   # create parameter grid
#   param_grid = linspace(param_lower, param_upper, param_N)
#
#   # create object to store moments at each parameter value
#   moment_storage = Array{Array{Float64}}(param_N)
#
#   # copy parameters structure to modify
#   paramsdec_float = deepcopy(paramsdec)
#   paramsshock_float = deepcopy(paramsshock)
#
#   # for each parameter guess, solve model and calculate full set of moments and store
#   for n in 1:param_N
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsprefs_float = ParametersPrefs(sigma1=param_grid[n])
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "sigma1")
#
# end
#
# function vary_sigma2(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
#
#   # create parameter grid
#   param_grid = linspace(param_lower, param_upper, param_N)
#
#   # create object to store moments at each parameter value
#   moment_storage = Array{Array{Float64}}(param_N)
#
#   # copy parameters structure to modify
#   paramsdec_float = deepcopy(paramsdec)
#   paramsshock_float = deepcopy(paramsshock)
#
#   # for each parameter guess, solve model and calculate full set of moments and store
#   for n in 1:param_N
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsprefs_float = ParametersPrefs(sigma2=param_grid[n])
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "sigma2")
#
# end
#
# function vary_rho12(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
#
#   # create parameter grid
#   param_grid = linspace(param_lower, param_upper, param_N)
#
#   # create object to store moments at each parameter value
#   moment_storage = Array{Array{Float64}}(param_N)
#
#   # copy parameters structure to modify
#   paramsdec_float = deepcopy(paramsdec)
#   paramsshock_float = deepcopy(paramsshock)
#
#   # for each parameter guess, solve model and calculate full set of moments and store
#   for n in 1:param_N
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsprefs_float = ParametersPrefs(rho12=param_grid[n])
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "rho12")
#
# end
#
# function vary_sigmab(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsshock_float = ParametersShock(eps_b_var=param_grid[n])
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "sigma2_b")
#
# end
#
# function vary_iota0(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsdec_float.iota0 = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "iota0")
#
# end
#
# function vary_iota1(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsdec_float.iota1 = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "iota1")
#
# end
#
# function vary_iota2(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsdec_float.iota2 = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "iota2")
#
# end
#
# function vary_iota3(initial_state_data,
#   paramsprefs::ParametersPrefs, paramsshock::ParametersShock, paramsdec::ParametersDec,
#   param_lower::Float64, param_upper::Float64, param_N::Int64;
#   N=1000, T=2, restrict_flag=1, S=100, seed=1234, sample_code="nodraw")
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
#     println(string("Iter ",n," of ",param_N,": ",param_grid[n]))
#     paramsdec_float.iota3 = param_grid[n]
#
#     # simulate dataset and compute moments
#     sim_moments = dgp_moments(initial_state_data, paramsprefs_float, paramsdec_float, paramsshock_float,
#       T=T, seed=seed, S=S, N=N, sample_code=sample_code, restrict_flag=1)
#
#     # store moments
#     moment_storage[n] = sim_moments[1]
#   end
#
#   # graph each moment
#   plot_moments(param_grid, moment_storage, "iota3")
#
# end
