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

test_params = [5.1220703125, 2.55859375, 0.559814453125, 0.177109375, 1.1748046875, 1.5849609375, 1.6728515625,
1.8876953125, 1.6142578125, 1.0185546875, 1.4072265625, 1.2060546875, 1.0361328125, 1.1845703125, 1.2685546875,
0.01775810546875, -1.08203125, 0.87700927734375, 0.47368544921875, -0.04296875]

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

last_params = [7.60223, 1.14087, 0.954799, 0.0210303, 1.87128, 1.17535, 1.29425, 1.64447, 1.88226, 1.32819,
1.96869, 1.46881, 1.33258, 1.75702, 1.48566, 0.00948392, -1.27173, 0.800475, 0.012733, -1.78052]

@elapsed test_smm_obj = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock)

@elapsed test_smm_obj_par = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

#= Investigate Parameter Vector That Didn't Converge In Bellman =#

trouble_params = [9.16504, 3.63672, 0.982822, 0.104922, 1.64746, 1.97168, 1.70801, 1.4541, 1.31348, 1.13184,
  1.74707, 1.75684, 1.71191, 1.44629, 1.30371, 0.0360991, -0.847656, 0.572337, 0.961918, -1.12109]
