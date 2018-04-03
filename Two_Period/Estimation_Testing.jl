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

# current minimum of 10k Sobol run

sobol_10k_min = [5.659027099609375, 3.6702880859375, 0.6154568481445313, 0.03391357421875, 1.406951904296875,
1.921356201171875, 1.374237060546875, 1.590667724609375, 1.106353759765625, 1.566314697265625, 1.654327392578125,
1.470672607421875, 1.882171630859375, 1.923919677734375, 1.025665283203125, 0.019858663940429688, 0.9503173828125,
0.9238209014892578, 0.28959131164550783, -0.3951416015625]

# starting at above, SMM optimizer

smm_1e1_min = [5.806976782509538, 3.596427798971455, 0.6474273333690911, 0.024909619904938913, 1.3236794681134911,
1.8960727698308322, 1.4290882930694442, 2.131879972007936, 1.0912259022687885, 1.7622338833900355, 1.5734799521702898,
1.555031106236331, 2.0759002188654074, 1.9994496447902468, 1.0532153752901494, 0.015112078345713015, 0.862103429945217,
0.8963280834533058, 0.31855282009330094, -0.39046389594752096]

@elapsed smm_obj_par = smm_obj_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

# read moments more easily

[smm_obj_par[3][1][1:10] smm_obj_par[3][2][1:10] smm_obj_par[4][1][1:10]]

[smm_obj_par[3][1][11:20] smm_obj_par[3][2][11:20] smm_obj_par[4][1][11:20]]

[smm_obj_par[3][1][21:30] smm_obj_par[3][2][21:30] smm_obj_par[4][1][21:30]]

[smm_obj_par[3][1][31:39] smm_obj_par[3][2][31:39] smm_obj_par[4][1][31:39]]

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

@elapsed test_smm_obj_par1 = smm_obj_testing(nlsy79data_formatted, last_params, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

#= Investigate Parameter Vector That Didn't Converge In Bellman =#

trouble_params = [9.16504, 3.63672, 0.982822, 0.104922, 1.64746, 1.97168, 1.70801, 1.4541, 1.31348, 1.13184,
  1.74707, 1.75684, 1.71191, 1.44629, 1.30371, 0.0360991, -0.847656, 0.572337, 0.961918, -1.12109]
