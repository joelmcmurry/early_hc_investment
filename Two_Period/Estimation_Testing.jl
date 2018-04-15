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

test_params = [0.1, 0.1, 0.95,
-270., -10.,
400., 100.,
400., 20.,
40., 10.,
0.4,
2.194, 0.1, 0.125, 0.02]

test_params = deepcopy(test_params)

paramsprefs = ParametersPrefs(sigma_alphaT1=test_params[1], sigma_alphaT2=test_params[2], rho=test_params[3])

paramsprefs.gamma_0 = [test_params[4], test_params[5]]
paramsprefs.gamma_y = [test_params[6], test_params[7]]
paramsprefs.gamma_a = [test_params[8], test_params[9]]
paramsprefs.gamma_b = [test_params[10], test_params[11]]
paramsshock.eps_b_var = test_params[12]

paramsdec.iota0 = test_params[13]
paramsdec.iota1 = test_params[14]
paramsdec.iota2 = test_params[15]
paramsdec.iota3 = test_params[16]

test_paths = sim_paths(initial_state_data, paramsshock, paramsprefs, seed=1234, N=5000, type_N=20)

plot_density(test_paths[2][:,1], "alphaT1", minimum(test_paths[2][:,1]), maximum(test_paths[2][:,1]))

plot_density(test_paths[2][:,2], "alphaT2", minimum(test_paths[2][:,2]), maximum(test_paths[2][:,2]))

@elapsed test_choices = sim_choices(test_paths[1], test_paths[2], test_paths[3], test_paths[4],
  paramsprefs, paramsdec, paramsshock, bellman_tol=1e-9, bellman_iter=5000, error_log_flag=0)

test_mom = moment_gen_dist(test_choices)

data_mom = moment_gen_dist(nlsy79data_formatted)

test_mom1 = [data_mom[1][1:7] test_mom[2][1:7] test_mom[1][1:7]]

test_mom2 = [data_mom[1][8:13] test_mom[2][8:13] test_mom[1][8:13]]

test_mom3 = [data_mom[1][14:25] test_mom[2][14:25] test_mom[1][14:25]]

test_mom4 = [data_mom[1][26:37] test_mom[2][26:37] test_mom[1][26:37]]

## Individual State/Pref Testing

n=1

# y_test = nlsy79data_formatted[1][1][n]
# a_test = nlsy79data_formatted[2][1][n]
# b_test = nlsy79data_formatted[3][1][n]

y_test = test_choices[1][1][n]
a_test = test_choices[2][1][n]
b_test = test_choices[3][1][n]

println(y_test, " ", a_test, " ", b_test)

mean_prefs = type_construct(y_test, a_test, b_test, paramsprefs, mean_flag=1, type_N=10)

test_alphaT1 = mean_prefs[1][1]
test_alphaT2 = mean_prefs[1][2]

test_alphaT1 = test_paths[2][n,1]
test_alphaT2 = test_paths[2][n,2]

# test_alphaT1 = 0.000000001
# test_alphaT2 = test_paths[2][n,2]

paramsdec.alphaT1 = test_alphaT1
paramsdec.alphaT2 = test_alphaT2

choices = bellman_optim_child!(y_test, a_test, b_test, paramsdec, paramsshock)

test_choices[2][2][n]
test_choices[5][1][n]

# aprime_actual = nlsy79data_formatted[2][2][n]
# s_actual = nlsy79data_formatted[4][1][n]
# x_actual = nlsy79data_formatted[5][1][n]

bT_noshock = HC_prod(b_test, choices[2][2], 0., paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3)

t_opt(Y_evol(y_test, paramsdec.rho_y, 0.), choices[2][1], bT_noshock,
  paramsdec.alphaT1, paramsdec.alphaT2, paramsdec.beta0, paramsdec.beta1, paramsdec.beta2, paramsdec.r)

# vary params

choices_vary_param("alphaT1", y_test, a_test, b_test, paramsdec, paramsshock, 1., 100., param_N=50)

#= Testing Moment Generation =#

@elapsed test_mom = dgp_moments(initial_state_data, paramsprefs, paramsdec, paramsshock, type_N=2, N=1000)

@elapsed test_mom_par = dgp_moments_par(initial_state_data, paramsprefs, paramsdec, paramsshock, par_N=4, type_N=2, N=1000)

#= Testing Sobol SMM/Write =#

@elapsed smm_sobol_write_results("sobol_test.txt", "sobol_store.csv", "sobol_store_error.csv", nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  sobol_N=10, par_flag=0, par_N=4, print_flag=0)

#= Testing SMM Objective Function or Particular Parameter Vector =#

# SMM optimizer

sobol_100k = [2.8229872169494628, 4.489028148651123, -0.8287337493896484, 0.09395217895507812,
-0.12639923095703132, 0.06224136352539064, -7.6587677001953125, -0.061412811279296875,
4.178622055053711, -0.06649246215820312, 0.1873065948486328, 0.811538470840454, 1.0388336181640625,
1.4026711643218994, 0.4082890048980713, -0.1326141357421875]

@elapsed smm_obj_par = smm_obj_testing(nlsy79data_formatted, sobol_10k_moretype, paramsprefs, paramsdec, paramsshock,
  par_flag=1, par_N=4, type_N=20, N=5000, pref_only_flag=0, error_log_flag=1)

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
  sobol_N=10, par_flag=1, par_N=4, print_flag=0)

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

#= Testing DGP with Serial and Parallel =#

mom_ser = dgp_moments(nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  seed=1234, N=100, restrict_flag=1, error_log_flag=1, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)

mom_par = dgp_moments_par(nlsy79data_formatted, paramsprefs, paramsdec, paramsshock,
  seed=1234, N=100, restrict_flag=1, par_N=2, error_log_flag=1, type_N=2, bellman_trace=false, bellman_iter=5000, bellman_tol=1e-9)
