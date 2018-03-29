#= Estimation of Parameters of Full Model =#

using DataFrames
using PyPlot

include("Utilities_Estimation.jl")

## Read in NLSY79 Data with Family Characteristics and Choices

nlsy79data = readtable("C:/Users/j0el/Documents/Wisconsin/Field Paper 2/R/Output/nlsy79_multi_period_est_nonneg.csv", header=true)

# transform data to form readable by moment generation function

nlsy79data_formatted = data_transform(nlsy79data,4)

## Initialize Parameters and Parameters Guess

param_vec = [0.902651, 0.463218, 8.58724, 0.999691, 0.253216, 0.0228707, 2.29149, 1.86123, 0.888602,
            0.516928, 0.452178, 0.56109, 0.052591, 0.0541078, 0.0975748, -0.000497188, 0.0101587, 0.0165175]

param_vec = [0.75858, 0.508858, 7.52927, 1.01505, 0.265082, 0.0252814, 1.96198, 1.97818, 0.884681,
            0.442704, 0.452736, 0.697661, 0.0648002, 0.0612778, 0.0976522, 0.00269696, 0.00478811, 0.0221447]

param_vec_temp_low_sd = [0.771292, 0.419102, 8.42858, 1.00624, 0.294094, 0.0428306, 1.86032, 2.21569,
            1.20663, 0.469206, 0.457585, 1.03107, 0.0547181, 0.0684785, 0.0215771, -0.015377, 0.000457233, -0.000616096]

param_vec_new_low_sd = [0.771271, 0.419049, 8.42964, 1.00624, 0.294212, 0.0428336, 1.86073, 2.21592,
            1.20603, 0.469789, 0.457523, 1.03077, 0.0547179, 0.0684213, 0.0215422, -0.0153543, 0.000432051, -0.000604273]

# e-1
param_vec = [0.771281, 0.419034, 8.42937, 1.00624, 0.294224, 0.0428329, 1.86094, 2.21601,
            1.20586, 0.469789, 0.457557, 1.03073, 0.0547358, 0.0684063, 0.0215387, -0.015361, 0.000437926, -0.00060102]

# e-4
param_vec = [0.771281, 0.419034, 8.42937, 1.00624, 0.294225, 0.042833, 1.86094, 2.21601,
            1.20586, 0.469787, 0.457557, 1.03073, 0.0547353, 0.0684066, 0.0215394, -0.0153606, 0.00043784, -0.000601216]

paramst = Parameterst(alpha01=param_vec[1], alpha02=1-param_vec[1],
              alpha1=param_vec[2], alpha2=1-param_vec[2], B=param_vec[3],
              rho_y=param_vec[4], eps_y_var=param_vec[5], eps_b_var=param_vec[6],
              iota0=param_vec[7:9], iota1=param_vec[10:12], iota2=param_vec[13:15],
              iota3=param_vec[16:18], y_N=[5 5 5 5], a_N=[5 5 5 5], b_N=[5 20 20 5],
              b_max=[50. 400. 400. 200.])

# test moment match

test_moment = smm_obj_testing(nlsy79data_formatted, param_vec, paramst, error_log_flag=2)

#= Estimation =#

smm_est = smm_all(nlsy79data_formatted, paramst, error_log_flag=2)

#= Backward Induction Testing =#

tic()
V_array, s_array, x_array, error_log1, error_log2 = back_induction_Optim(paramst, 4, error_log_flag=1)
toc()

# V Plots

V_Tfig = figure()
plot(linspace(1,paramst.state_N[4],paramst.state_N[4]), V_array[4], label="V_T")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

V_3fig = figure()
plot(linspace(1,paramst.state_N[3],paramst.state_N[3]), V_array[3], label="V_3")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

V_2fig = figure()
plot(linspace(1,paramst.state_N[2],paramst.state_N[2]), V_array[2], label="V_2")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

V_1fig = figure()
plot(linspace(1,paramst.state_N[1],paramst.state_N[1]), V_array[1], label="V_1")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

# s Plots

s_3fig = figure()
plot(linspace(1,paramst.state_N[3],paramst.state_N[3]), s_array[3], label="s_3")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

s_2fig = figure()
plot(linspace(1,paramst.state_N[2],paramst.state_N[2]), s_array[2], label="s_2")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

s_1fig = figure()
plot(linspace(1,paramst.state_N[1],paramst.state_N[1]), s_array[1], label="s_1")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

# x Plots

x_3fig = figure()
plot(linspace(1,paramst.state_N[3],paramst.state_N[3]), x_array[3], label="x_3")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

x_2fig = figure()
plot(linspace(1,paramst.state_N[2],paramst.state_N[2]), x_array[2], label="x_2")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

x_1fig = figure()
plot(linspace(1,paramst.state_N[1],paramst.state_N[1]), x_array[1], label="x_1")
ax = PyPlot.gca()
ax[:legend](loc="lower right")

# old manual induction not sure why i have this

# solve last period on last period grid
V_T = fill(-Inf, paramst.state_N[4])
for i in 1:paramst.state_N[4]
  V_T[i] = EV_T(paramst.state_grid4[i,1], paramst.state_grid4[i,2], paramst.state_grid4[i,3], paramst)
end

V_3 = fill(-Inf, paramst.state_N[3])
s_3 = zeros(paramst.state_N[3])
x_3 = zeros(paramst.state_N[3])
for i in 1:paramst.state_N[3]
  opt_sol = bellman_Optim_Tminus1!(paramst.state_grid3[i,1], paramst.state_grid3[i,2], paramst.state_grid3[i,3], paramst)
  V_3[i] = opt_sol[1]
  s_3[i] = opt_sol[2][1]
  x_3[i] = opt_sol[2][2]
end

V_2 = fill(-Inf, paramst.state_N[2])
s_2 = zeros(paramst.state_N[2])
x_2 = zeros(paramst.state_N[2])
for i in 1:paramst.state_N[2]
  opt_sol = bellman_Optim_interp!(paramst.state_grid2[i,1], paramst.state_grid2[i,2], paramst.state_grid2[i,3],
    2, V_3, paramst.state_grid_interp3, paramst)
  V_2[i] = opt_sol[1]
  s_2[i] = opt_sol[2][1]
  x_2[i] = opt_sol[2][2]
end

V_1 = fill(-Inf, paramst.state_N[1])
s_1 = zeros(paramst.state_N[1])
x_1 = zeros(paramst.state_N[1])
for i in 1:paramst.state_N[1]
  opt_sol = bellman_Optim_interp!(paramst.state_grid1[i,1], paramst.state_grid1[i,2], paramst.state_grid1[i,3],
    1, V_2, paramst.state_grid_interp2, paramst)
  V_1[i] = opt_sol[1]
  s_1[i] = opt_sol[2][1]
  x_1[i] = opt_sol[2][2]
end
