#= Estimation Testing =#

addprocs(4)

@everywhere include("Utilities_Estimation_2P.jl")

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data)

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

#= Testing Moment Generation =#

initial_state_data = nlsy79data_formatted

@elapsed test_mom = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock)

@elapsed test_mom_par = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock, par_N=4)

#= Testing SMM Objective Function or Particular Parameter Vector =#

test_params = [9.62646484375, 2.052734375, 0.9777978515625, 0.454345703125, 0.14277343750000002,
  1.49169921875, 1.08837890625, 1.66943359375, 1.72998046875, 1.96435546875, 1.24951171875, 1.28271484375,
  1.40380859375, 1.50634765625, 1.66943359375, 1.41650390625, 1.53564453125, 1.73388671875, 1.62158203125,
  1.51806640625, 1.31103515625, 1.09228515625, 1.68017578125, 0.088927099609375, -0.287109375,
  1.163127783203125, 0.495655908203125, -0.150390625]

@elapsed test_smm_obj = smm_obj_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock)

@elapsed test_smm_obj_par = smm_obj_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

#= Testing Sobol SMM/Write =#

@elapsed smm_sobol_write_results("sobol_test.txt", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=1, par_flag=1, par_N=4, print_flag=0)

#= TEMP TESTING DGP =#

using DataFrames

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

initial_state_data = data_transform(nlsy79data)

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

test_paths = sim_paths(initial_state_data, paramsshock)

initial_states = test_paths[1]

shocks_y = test_paths[2][1]
shocks_b = test_paths[3][1]

test_choices = sim_choices(initial_states, shocks_y, shocks_b, paramsprefs, paramsdec, paramsshock)

#= Troubleshooting Sobol Linstat Run =#

last_params = [2.12665, 4.56665, 0.561938, 0.351593, 0.134966, 1.9361, 1.51398,
  1.9267, 1.55902, 1.52057, 1.39154, 1.38385, 1.8277, 1.31268, 1.64874, 1.48749, 1.55804,
  1.10822, 1.71088, 1.61249, 1.88483, 1.92975, 1.64838, 0.0413734, 1.36646, 0.889094, 0.161766, 0.247314]

@elapsed test_smm_obj = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock)

@elapsed test_smm_obj_par = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)
