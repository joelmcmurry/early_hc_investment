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

last_params = [3.06048583984375, 2.096923828125, 0.7017840576171874, 0.672271728515625, 0.0374072265625,
  1.84844970703125, 1.08648681640625, 1.21697998046875, 1.44500732421875, 1.41339111328125, 1.98358154296875,
  1.14483642578125, 1.66925048828125, 1.37054443359375, 1.14508056640625, 1.30072021484375, 1.90789794921875,
  1.86773681640625, 1.52410888671875, 1.41546630859375, 1.00567626953125, 1.50421142578125, 1.02410888671875,
  0.08651862182617188, -0.466552734375, 0.06638086547851563, 0.8025710144042969, -0.720458984375]

@elapsed test_smm_obj = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock)

@elapsed test_smm_obj_par = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

#= Investigate Parameter Vector That Didn't Converge In Bellman =#

trouble_params = [9.16504, 3.63672, 0.982822, 0.104922, 1.64746, 1.97168, 1.70801, 1.4541, 1.31348, 1.13184,
  1.74707, 1.75684, 1.71191, 1.44629, 1.30371, 0.0360991, -0.847656, 0.572337, 0.961918, -1.12109]
