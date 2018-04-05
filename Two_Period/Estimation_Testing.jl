#= Estimation Testing =#

#= Fix PyPlot Issue =#

ENV["MPLBACKEND"]="qt4agg"

#= End =#

addprocs(4)

@everywhere include("Utilities_Estimation_2P.jl")

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data)

initial_state_data = nlsy79data_formatted

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

#= Test DGP =#

test_paths = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=1234, N=1000)

@elapsed test_choices = sim_choices(test_paths[1], test_paths[2], test_paths[3], test_paths[4], paramsprefs, paramsdec, paramsshock)

test_mom = moment_gen_dist(test_choices)

#= Testing Moment Generation =#

initial_state_data = nlsy79data_formatted

@elapsed test_mom = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock)

@elapsed test_mom_par = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock, par_N=4)

#= Testing SMM Objective Function or Particular Parameter Vector =#

# SMM optimizer

smm_1e1_min = [7.440572977878828, 3.1714410121105088, 0.6138021146248615, 0.05978069786455791,
1.3517041360143274, 1.435700915256765, 1.3325181388182585,
2.1748539567190504, 0.7218463209945897, 1.5594413250003598, 1.983579896628608,
2.0016641984824037, 1.0043868128423774, 1.9383126352689801, 1.0755957319412595,
0.03114075090791482, 1.676017468842615, 0.10041386658424886, 0.05408924598776115, 0.04679703798950916]

@elapsed smm_obj_par = smm_obj_testing(nlsy79data_formatted, smm_1e1_min, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

# read moments more easily

mom_comp1 = [smm_obj_par[3][1][1:9] smm_obj_par[3][2][1:9] smm_obj_par[4][1][1:9]]

mom_comp2 = [smm_obj_par[3][1][10:15] smm_obj_par[3][2][10:15] smm_obj_par[4][1][10:15]]

mom_comp3 = [smm_obj_par[3][1][16:27] smm_obj_par[3][2][16:27] smm_obj_par[4][1][16:27]]

mom_comp4 = [smm_obj_par[3][1][28:39] smm_obj_par[3][2][28:39] smm_obj_par[4][1][28:39]]

#= Testing Sobol SMM/Write =#

@elapsed smm_sobol_write_results("sobol_test.txt", "sobol_store.csv", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=50, par_flag=1, par_N=4, print_flag=0)

#= Test Moment Sensitivity =#

@elapsed B_hi_test = vary_param("B_hi", nlsy79data_formatted, smm_1e1_min_mod, paramsprefs, paramsshock, paramsdec, 3.6, 10., 10, par_flag=1, par_N=4)

plot_moments(B_hi_test[1], B_hi_test[2], B_hi_test[3], moment_display_flag=1)

@elapsed type_test = vary_param("gamma_y2", nlsy79data_formatted, smm_1e1_min_mod, paramsprefs, paramsshock, paramsdec, 0., 3., 10, par_flag=1, par_N=4)

plot_moments(type_test[1], type_test[2], type_test[3], moment_display_flag=2)

#= "Indetifiation" =#

# run and store sobol sequence tests

@elapsed smm_sobol_write_results("sobol_test.txt", "sobol_store.csv", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=100, par_flag=1, par_N=4, print_flag=0)

# compute data moments

data_moments = moment_gen_dist(nlsy79data_formatted)[1]

# plot all moments given parameter

parameter_index = 1

hold_param_constant = param_constant_quantile(parameter_index, "sobol_store.csv"; bin_N=10)

plot_moment_quantiles_all_moments(hold_param_constant[1], hold_param_constant[2], hold_param_constant[3], hold_param_constant[4], hold_param_constant[5], data_moments)

# plot all parameters given a single moment

moment_index = 1

plot_moment_quantiles_all_params(moment_index, data_moments, "sobol_store.csv", bin_N=10)


test_norm = rand(Normal(0,1))

pdf(Normal(0,1),test_norm)

test_mv = rand(MvNormal(eye(2)),2)

pdf(MvNormal(eye(2)), [test_mv[1]; test_mv[2]])
