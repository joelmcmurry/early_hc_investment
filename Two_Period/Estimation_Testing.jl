#= Estimation Testing =#

addprocs(4)

@everywhere include("Utilities_Estimation_2P.jl")

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data)

initial_state_data = nlsy79data_formatted

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

@elapsed test_mom = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock)

@elapsed test_mom_par = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock, par_N=4)

test_params = [0.359929, 0.755052, 0.11252, 0.0597615, 0.224423, 0.0240061, 1.65583, 0.316465, 0.114916, 0.0154898]

@elapsed test_smm_obj = smm_obj_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock)

@elapsed test_smm_obj_par = smm_obj_testing(nlsy79data_formatted, test_params, paramsprefs, paramsdec, paramsshock, par_flag=1, par_N=4)
