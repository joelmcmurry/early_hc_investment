#= Estimation of Parameters of Full Model =#

#= Fix PyPlot Issue =#

ENV["MPLBACKEND"]="qt4agg"

#= End =#

addprocs(4)

using DataFrames
using PyPlot

@everywhere include("Utilities_Estimation_2P.jl")

## Read in NLSY79 Data with Family Characteristics and Choices

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

# transform data to form readable by moment generation function

nlsy79data_formatted = data_transform(nlsy79data)

## Initialize Parameters and Parameters Guess

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

param_guess = [0.53 0.81 0.21 0.99 0.81 0.54 -0.83 0.89 0.43 -0.06]

## Test SMM Ojective Function

test_obj_smm = smm_obj_testing(nlsy79data_formatted, param_guess, paramsprefs, paramsdec, paramsshock, par_flag=1, par_N=4)

## Test Sobol Sequence SMM and Write to Output File

smm_sobol_write_results("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/sobol_test.txt",
   nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, sobol_N=10, par_flag=1, par_N=4)

#= Identification Investigation =#

vary_mu1(nlsy79data_formatted, paramsprefs, paramsshock, paramsdec, 0.1, 0.9, 10, sample_code="nodraw")

#= Estimation =#

est_params = smm(nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, opt_trace=true, restrict_flag=1,
   mu1_start=param_guess[1], mu2_start=param_guess[2], sigma1_start=param_guess[3], sigma2_start=param_guess[4],
   rho12_start=param_guess[5], eps_b_var_start=param_guess[6],
   iota0_start=param_guess[7], iota1_start=param_guess[8], iota2_start=param_guess[9], iota3_start=param_guess[10], opt_iter=2, par_flag=1, par_N=4)
