#= Estimate Parameters on Sobol Sequence =#

addprocs(8)

using DataFrames

# include estimation and solution utilities

println("it's working")

@everywhere include("/home/m/mcmurry2/Utilities_Estimation_2P.jl")

## Read in NLSY79 Data with Family Characteristics and Choices

nlsy79data = readtable("/home/m/mcmurry2/two_period_states_controls_for_est.csv", header=true)

# transform data to form readable by moment generation function

@everywhere nlsy79data_formatted = data_transform(nlsy79data)

## Initialize Parameters

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

## Carry Out SMM Estimation on Sobol Sequence and Write to Output File

smm_sobol_write_results("/home/m/mcmurry2/sobol_test.txt", "/home/m/mcmurry2/sobol_store.csv",
nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, sobol_N=10, par_flag=1, par_N=8, error_log_flag=1)
