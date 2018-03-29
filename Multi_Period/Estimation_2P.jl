#= Estimation of Parameters of Full Model =#

using DataFrames
using PyPlot

include("Utilities_Estimation_2P.jl")

## Read in NLSY79 Data with Family Characteristics and Choices

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Field Paper 2/R/Output/nlsy79_multi_period_est_nonneg_alt2.csv", header=true)

# transform data to form readable by moment generation function

nlsy79data_formatted = data_transform(nlsy79data)

## Initialize Parameters and Parameters Guess

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

param_prefs_guess = [0.3, 0.3, 0.1, 0.1, 0.25, 0.5, 1.0]

param_all_guess = [0.3, 0.3, 0.1, 0.1, 0.25, 0.5, 1.0, 0.997, 0.251, 0.022, 1.87, 0.42, 0.06, 0.0]

#= Estimation =#

est_prefs = smm_prefs(nlsy79data_formatted, paramsprefs, paramsdec, paramsshock, opt_trace=true,
  restrict_flag=1)
