#= Estimation Testing =#

#= Fix PyPlot Issue =#

ENV["MPLBACKEND"]="qt4agg"

#= End =#

addprocs(4)

@everywhere include("Utilities_Estimation_2P.jl")

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data)

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

#= Test DGP =#

initial_state_data = nlsy79data_formatted

test_paths = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=1234, N=10000)

@elapsed test_choices = sim_choices(test_paths[1], test_paths[2], test_paths[3], test_paths[4], paramsprefs, paramsdec, paramsshock)

test_mom = moment_gen_dist(test_choices)

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

@elapsed smm_obj_par = smm_obj_testing(nlsy79data_formatted, smm_1e1_min, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4)

# read moments more easily

mom_comp1 = [smm_obj_par[3][1][1:9] smm_obj_par[3][2][1:9] smm_obj_par[4][1][1:9]]

mom_comp2 = [smm_obj_par[3][1][10:15] smm_obj_par[3][2][10:15] smm_obj_par[4][1][10:15]]

mom_comp3 = [smm_obj_par[3][1][16:27] smm_obj_par[3][2][16:27] smm_obj_par[4][1][16:27]]

mom_comp4 = [smm_obj_par[3][1][28:39] smm_obj_par[3][2][28:39] smm_obj_par[4][1][28:39]]

#= Testing Sobol SMM/Write =#

@elapsed smm_sobol_write_results("sobol_test.txt", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=1, par_flag=1, par_N=4, print_flag=0)

#= Test Moment Sensitivity =#

B_hi_test = vary_B_hi(nlsy79data_formatted, smm_1e1_min, paramsprefs, paramsshock, paramsdec, 3.6, 10., 10)
