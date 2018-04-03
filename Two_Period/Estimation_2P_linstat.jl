#= Estimate Parameters =#

addprocs(8)

using DataFrames

# include estimation and solution utilities

println("it's working")

@everywhere include("/home/m/mcmurry2/Utilities_Estimation_2P.jl")

## Read in NLSY79 Data with Family Characteristics and Choices

nlsy79data = readtable("/home/m/mcmurry2/two_period_states_controls_for_est.csv", header=true)

# transform data to form readable by moment generation function

nlsy79data_formatted = data_transform(nlsy79data)

## Initialize Parameters and Parameter Guesss

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

param_guess = [5.659027099609375, 3.6702880859375, 0.6154568481445313, 0.03391357421875, 1.406951904296875,
 1.921356201171875, 1.374237060546875, 1.590667724609375, 1.106353759765625, 1.566314697265625, 1.654327392578125,
 1.470672607421875, 1.882171630859375, 1.923919677734375, 1.025665283203125, 0.019858663940429688, 0.9503173828125,
 0.9238209014892578, 0.28959131164550783, -0.3951416015625]

## Carry Out SMM Estimation and Write to Output File

smm_write_results("/home/m/mcmurry2/smm_test.txt", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, restrict_flag=1,
 B_hi_start=param_guess[1], B_lo_start=param_guess[2], alphaT1_hi_start=param_guess[3], alphaT1_lo_start=param_guess[4],
 gamma_y2_start=param_guess[5], gamma_y3_start=param_guess[6], gamma_y4_start=param_guess[7],
 gamma_a1_start=param_guess[8], gamma_a2_start=param_guess[9], gamma_a3_start=param_guess[10], gamma_a4_start=param_guess[11],
 gamma_b1_start=param_guess[12], gamma_b2_start=param_guess[13], gamma_b3_start=param_guess[14], gamma_b4_start=param_guess[15],
 eps_b_var_start=param_guess[16], iota0_start=param_guess[17], iota1_start=param_guess[18], iota2_start=param_guess[19], iota3_start=param_guess[20],
 opt_trace=true, par_flag=1, par_N=8, opt_tol=1e-1)
