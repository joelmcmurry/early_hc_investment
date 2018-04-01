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

# param_guess = [0.31 0.76 0.10 0.07 0.16 0.04 1.8 0.19 0.14 0.03]

## Carry Out SMM Estimation and Write to Output File

smm_write_results("/home/m/mcmurry2/smm_test.txt", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, restrict_flag=1,
 mu1_start=param_guess[1], mu2_start=param_guess[2], sigma1_start=param_guess[3], sigma2_start=param_guess[4],
 rho12_start=param_guess[5], eps_b_var_start=param_guess[6],
 iota0_start=param_guess[7], iota1_start=param_guess[8], iota2_start=param_guess[9], iota3_start=param_guess[10],
 opt_trace=true, par_flag=1, par_N=8, opt_tol=1e-1)
