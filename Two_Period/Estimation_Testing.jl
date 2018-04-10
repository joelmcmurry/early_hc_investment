#= Estimation Testing =#

#= Fix PyPlot Issue =#

ENV["MPLBACKEND"]="qt4agg"

#= End =#

addprocs(4)

@everywhere include("Utilities_Estimation_2P.jl")

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Projects/Early_HC_Investment/R/Output/two_period_states_controls_for_est.csv", header=true)

nlsy79data_formatted = data_transform(nlsy79data)

initial_state_data = nlsy79data_formatted

paramsdec = ParametersDec()
paramsprefs = ParametersPrefs()
paramsshock = ParametersShock()

#= Test DGP and Tinker with Parameters =#

test_params = [1., 1., -0.5,
.1, -1.,
3.1, -7.5,
.1, -5.1,
.15, -2.1,
0.4,
2.6, 0.25, 0.062, 0.]

sobol_result_pref_only = [1.053558349609375, 1.910491943359375, -0.9333819580078125, 0.030084228515625006,
0.032171630859374994, 0.07411499023437501, 0.06156616210937502, 0.068914794921875, -0.076385498046875,
0.073419189453125, -0.080853271484375,
0.022, 2.97, 0.27, 0.02, 0.]

sobol_25k = [1.303497314453125, 1.089263916015625, -0.857647705078125, -0.030355834960937494,
0.00042114257812500555, -0.0788238525390625, 0.0004943847656249944, 0.13460388183593752, 0.085638427734375,
0.06196594238281253, -0.031182861328125006,
0.04096771240234375, -7.0474853515625, 0.6693145751953125, 1.73492462158203127, -0.224908447265625]

test_params = deepcopy(sobol_25k)

paramsprefs = ParametersPrefs(sigma_B=test_params[1], sigma_alphaT1=test_params[2], rho=test_params[3])

paramsprefs.gamma_0 = [test_params[4], test_params[5]]
paramsprefs.gamma_y = [test_params[6], test_params[7]]
paramsprefs.gamma_a = [test_params[8], test_params[9]]
paramsprefs.gamma_b = [test_params[10], test_params[11]]
paramsshock.eps_b_var = test_params[12]
paramsdec.iota0 = test_params[13]
paramsdec.iota1 = test_params[14]
paramsdec.iota2 = test_params[15]
paramsdec.iota3 = test_params[16]

test_paths = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=1234, N=500, type_N=6)

@elapsed test_choices = sim_choices(test_paths[1], test_paths[2], test_paths[3], test_paths[4],
  paramsprefs, paramsdec, paramsshock, bellman_tol=1e-9, bellman_iter=5000, error_log_flag=1)

test_mom = moment_gen_dist(test_choices)

data_mom = moment_gen_dist(nlsy79data_formatted)

test_mom1 = [data_mom[1][1:9] test_mom[2][1:9] test_mom[1][1:9]]

test_mom2 = [data_mom[1][10:15] test_mom[2][10:15] test_mom[1][10:15]]

test_mom3 = [data_mom[1][16:27] test_mom[2][16:27] test_mom[1][16:27]]

test_mom4 = [data_mom[1][28:39] test_mom[2][28:39] test_mom[1][28:39]]

## Individual State/Pref Testing

n=2

y_test = nlsy79data_formatted[1][1][n]
a_test = nlsy79data_formatted[2][1][n]
b_test = nlsy79data_formatted[3][1][n]
test_B = 200.
test_alphaT1 = 0.005
test_alphaT2 = 0.5

paramsdec_test = ParametersDec(B=test_B, alphaT1=test_alphaT1, iota2=0.11)

choices_vary_param("alphaT1", y_test, a_test, b_test, paramsdec_test, paramsshock, 0.001, 0.05, param_N=20)

aprime_actual = nlsy79data_formatted[2][2][n]
s_actual = nlsy79data_formatted[4][1][n]
x_actual = nlsy79data_formatted[5][1][n]

bellman_optim_child!(y_test, a_test, b_test, paramsdec_test, paramsshock)

# mean next-period income
y_annual_test = Y_evol(y_test, paramsdec.rho_y, 0.)

# mean next-period HC
bprime_test = HC_prod(b_test, x_actual, 0.,
  paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3)

t_opt_vary_parma("alphaT1", y_annual_test, aprime_actual, bprime_test, paramsdec_test, 0.001, 0.1, param_N=20)

#= Testing Moment Generation =#

@elapsed test_mom = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock, type_N=2, N=1000)

@elapsed test_mom_par = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock, par_N=4, type_N=2, N=1000)

#= Testing Sobol SMM/Write =#

@elapsed smm_sobol_write_results("sobol_test.txt", "sobol_store.csv", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=100, par_flag=1, par_N=4, print_flag=0)

#= Testing SMM Objective Function or Particular Parameter Vector =#

# SMM optimizer

sobol_1000 = [0.6559458007812501, 2.61522314453125, -0.965830078125, -0.07724609375000001, 0.05244140625000002,
0.02763671875000001, 0.05791015625000001, 0.0077148437500000056, 0.013574218750000006, 0.04638671875, -0.0028320312500000056]

@elapsed smm_obj_par = smm_obj_testing(nlsy79data_formatted, sobol_1000, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4, type_N=6, N=1000, pref_only_flag=1, error_log_flag=1)

# read moments more easily

mom_comp1 = [smm_obj_par[3][1][1:9] smm_obj_par[3][2][1:9] smm_obj_par[4][1][1:9]]

mom_comp2 = [smm_obj_par[3][1][10:15] smm_obj_par[3][2][10:15] smm_obj_par[4][1][10:15]]

mom_comp3 = [smm_obj_par[3][1][16:27] smm_obj_par[3][2][16:27] smm_obj_par[4][1][16:27]]

mom_comp4 = [smm_obj_par[3][1][28:39] smm_obj_par[3][2][28:39] smm_obj_par[4][1][28:39]]

#= Test Parameters Throwing Errors in SMM =#

param_trouble = [0.297727, 3.86741, 0.139219, -0.460938, 0.171875, 0.984375, -0.953125,
0.640625, 0.734375, 0.984375, -0.890625, 0.445368, -0.09375, 0.921929, 0.882824, -1.15625]

@elapsed smm_obj_trouble = smm_obj_testing(nlsy79data_formatted, param_trouble, paramsprefs, paramsdec, paramsshock,
  par_flag=0, par_N=4, error_log_flag=1)

#= "Indetifiation" =#

# run and store sobol sequence tests

@elapsed smm_sobol_write_results("sobol_test.txt", "sobol_store.csv", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=100, par_flag=1, par_N=4, print_flag=0)

# compute data moments

data_moments = moment_gen_dist(nlsy79data_formatted)[1]

# plot all moments given parameter

parameter_index = 3

hold_param_constant = param_constant_quantile(parameter_index, "sobol_store.csv"; bin_N=10)

plot_moment_quantiles_all_moments(hold_param_constant[1], hold_param_constant[2], hold_param_constant[3], hold_param_constant[4], hold_param_constant[5], data_moments)

# plot all parameters given a single moment

moment_index = 1

plot_moment_quantiles_all_params(moment_index, data_moments, "sobol_store.csv", bin_N=10)

#= Test Moment Sensitivity =#

@elapsed B_hi_test = vary_param("B_hi", nlsy79data_formatted, smm_1e1_min_mod, paramsprefs, paramsshock, paramsdec, 3.6, 10., 10, par_flag=1, par_N=4)

plot_moments(B_hi_test[1], B_hi_test[2], B_hi_test[3], moment_display_flag=1)

@elapsed type_test = vary_param("gamma_y2", nlsy79data_formatted, smm_1e1_min_mod, paramsprefs, paramsshock, paramsdec, 0., 3., 10, par_flag=1, par_N=4)

plot_moments(type_test[1], type_test[2], type_test[3], moment_display_flag=2)
