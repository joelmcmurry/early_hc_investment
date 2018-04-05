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

param_guess = [7.558196065778917, 2.470346590540029, 0.6151117540777155, 0.05069293030685955, 1.4013086367670247,
1.5180095820844521, 1.394453397265857, 2.21308939580944, 0.7067245747775206, 1.5920319249098498, 1.909234299152136,
1.9159362745785422, 1.0310501550991753, 1.9653493523428303, 1.0680806059811505, 0.029239918680193714,
1.53638450002311, 0.09959967336192384, 0.04719964718423235, 0.04839873962247803]


## Carry Out SMM Estimation and Write to Output File

smm_write_results("/home/m/mcmurry2/smm_test.txt", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, restrict_flag=1,
 B_hi_start=param_guess[1], B_lo_start=param_guess[2], alphaT1_hi_start=param_guess[3], alphaT1_lo_start=param_guess[4],
 gamma_y2_start=param_guess[5], gamma_y3_start=param_guess[6], gamma_y4_start=param_guess[7],
 gamma_a1_start=param_guess[8], gamma_a2_start=param_guess[9], gamma_a3_start=param_guess[10], gamma_a4_start=param_guess[11],
 gamma_b1_start=param_guess[12], gamma_b2_start=param_guess[13], gamma_b3_start=param_guess[14], gamma_b4_start=param_guess[15],
 eps_b_var_start=param_guess[16], iota0_start=param_guess[17], iota1_start=param_guess[18], iota2_start=param_guess[19], iota3_start=param_guess[20],
 opt_trace=true, par_flag=1, par_N=8, opt_tol=1e-1)
