#= Utilities for Solution of Model (Two-Period) =#

using Optim
using Distributions
using QuantEcon: gridmake

println("solution utilities loading")

#= Parameters Structures =#

## Decision Problem Parameters

mutable struct ParametersDec
  beta :: Float64 ## discount rate
  r :: Float64 ## interest rate
  B :: Float64 ## temporary continuation value shifter

  alphaT1 :: Float64 ## T-period utility parameter (RF choice)
  alphaT2 :: Float64 ## T-period utility parameter (RF choice)

  iota0 :: Float64 ## Vector of HC production function parameters
  iota1 :: Float64 ## Vector of HC production function parameters
  iota2 :: Float64 ## Vector of HC production function parameters
  iota3 :: Float64 ## Vector of HC production function parameters

  beta0 :: Float64 ## RF college quality parameter
  beta1 :: Float64 ## RF college quality parameter
  beta2 :: Float64 ## RF college quality parameter

  rho_y :: Float64 ## income persistence parameter

  function ParametersDec(;beta=0.5, r=0.18, B=1.,
    alphaT1=0.5, alphaT2=1-alphaT1,
    iota0=1.87428633, iota1=0.42122805,
    iota2=0.05979631, iota3=0.0,
    beta0=4.134, beta1=0.03, beta2=0.003, rho_y=0.997)

    new(beta, r, B, alphaT1, alphaT2, iota0, iota1, iota2, iota3, beta0, beta1, beta2, rho_y)

  end

end

## Shock Parameters

mutable struct ParametersShock
  eps_b_dist :: Distribution ## Distribution of HC shock
  eps_b_mu :: Float64 ## Mean of HC shock
  eps_b_var :: Float64 ## Variance of HC shock
  eps_b_discrete_min :: Float64 ## Minimum value of discretized distribution of HC shock
  eps_b_discrete_max :: Float64 ## Maximum value of discretized distribution of HC shock
  eps_b_discrete_N :: Int64 ## Number of discretized HC shock values
  eps_b_discrete_range :: Array ## Range of discretized HC shock values
  eps_b_dist_discrete :: Array ## PMF of discretized HC shock

  eps_y_dist :: Distribution ## Distribution of income shock
  eps_y_mu :: Float64 ## Mean of income shock
  eps_y_var :: Float64 ## Variance of income shock
  eps_y_discrete_min :: Float64 ## Minimum value of discretized distribution of income shock
  eps_y_discrete_max :: Float64 ## Maximum value of discretized distribution of income shock
  eps_y_discrete_N :: Int64 ## Number of discretized income shock values
  eps_y_discrete_range :: Array ## Range of discretized income shock values
  eps_y_dist_discrete :: Array ## PMF of discretized income shock

  eps_joint_discrete_N :: Int64 ## Number of discretized joint shock values
  eps_joint_discrete_range :: Array ## Range of discretized joint shock values
  eps_joint_dist_discrete :: Array ## PMF of discretized joint shock

  sd_num_b :: Float64 ## Number of SD to allow for shock values
  sd_num_y :: Float64 ## Number of SD to allow for shock values

  function ParametersShock(;eps_b_mu=0.0, eps_b_var=0.022, eps_b_discrete_N=10,
              eps_y_mu=0.0, eps_y_var=0.2513478, eps_y_discrete_N=10,
              sd_num_b=2.0, sd_num_y=2.0)

    # discretize b distribution

    eps_b_dist = Normal(eps_b_mu, eps_b_var)

    eps_b_discrete_min = mean(eps_b_dist) - sd_num_b*std(eps_b_dist)
    eps_b_discrete_max = mean(eps_b_dist) + sd_num_b*std(eps_b_dist)

    eps_b_discrete_range = linspace(eps_b_discrete_min,eps_b_discrete_max,eps_b_discrete_N)
    eps_b_dist_discrete = pdf.(eps_b_dist, eps_b_discrete_range)/sum(pdf.(eps_b_dist, eps_b_discrete_range))

    # discretize y distribution

    eps_y_dist = Normal(eps_y_mu, eps_y_var)

    eps_y_discrete_min = mean(eps_y_dist) - sd_num_y*std(eps_y_dist)
    eps_y_discrete_max = mean(eps_y_dist) + sd_num_y*std(eps_y_dist)

    eps_y_discrete_range = linspace(eps_y_discrete_min,eps_y_discrete_max,eps_y_discrete_N)
    eps_y_dist_discrete = pdf.(eps_y_dist, eps_y_discrete_range)/sum(pdf.(eps_y_dist, eps_y_discrete_range))

    # calculate discrete joint distribution

    eps_joint_discrete_range = gridmake(eps_y_discrete_range, eps_b_discrete_range)

    eps_joint_pre_mult = gridmake(eps_y_dist_discrete, eps_b_dist_discrete)

    eps_joint_dist_discrete = eps_joint_pre_mult[:,1].*eps_joint_pre_mult[:,2]

    eps_joint_discrete_N = eps_y_discrete_N*eps_b_discrete_N

    new(eps_b_dist, eps_b_mu, eps_b_var, eps_b_discrete_min, eps_b_discrete_max, eps_b_discrete_N,
      eps_b_discrete_range, eps_b_dist_discrete,
      eps_y_dist, eps_y_mu, eps_y_var, eps_y_discrete_min, eps_y_discrete_max, eps_y_discrete_N,
      eps_y_discrete_range, eps_y_dist_discrete,
      eps_joint_discrete_N, eps_joint_discrete_range, eps_joint_dist_discrete,
      sd_num_b, sd_num_y)

  end

end

## State Grids

mutable struct StateGrids
  y_min :: Array{Float64} ## income grid lower bound
  y_max :: Array{Float64} ## income grid upper bound
  y_N :: Array{Int64} ## number of income grid points
  y_grid :: Array ## income grid in t=1
  y_gridT :: Array ## income grid in t=T

  a_min :: Array{Float64} ## asset grid lower bound
  a_max :: Array{Float64} ## asset grid upper bound
  a_N :: Array{Int64} ## number of asset grid points
  a_grid :: Array ## asset grid in t=1
  a_gridT :: Array ## asset grid in t=2

  b_min :: Array{Float64} ## HC grid lower bound
  b_max :: Array{Float64} ## HC grid upper bound
  b_N :: Array{Int64} ## number of HC grid points
  b_grid :: Array ## HC grid in t=1
  b_gridT :: Array ## HC grid in t=T

  state_N :: Array{Int64} ## number of state grid points
  state_grid :: Array ## state grid in t=1
  state_gridT :: Array ## state grid in t=T

  function StateGrids(;y_min=[1000.0 1000.0], y_max=[150000.0 150000.0], y_N=[5 5],
              a_min=[1.0 1.0], a_max=[100000.0 100000.0], a_N=[5 5],
              b_min=[5.0 15.0], b_max=[60.0 70.0], b_N=[5 5])

    # create grids (uniform)

    # y_grid = linspace(y_min, y_max, y_N)
    # a_grid = linspace(a_min, a_max, a_N)
    # b_grid = linspace(b_min, b_max, b_N)

    # create grids (chebyshev spacing)

    y_grid = cheby_nodes!(y_min[1], y_max[1], y_N[1])
    y_gridT = cheby_nodes!(y_min[2], y_max[2], y_N[2])

    a_grid = cheby_nodes!(a_min[1], a_max[1], a_N[1])
    a_gridT = cheby_nodes!(a_min[2], a_max[2], a_N[2])

    b_grid = cheby_nodes!(b_min[1], b_max[1], b_N[1])
    b_gridT = cheby_nodes!(b_min[2], b_max[2], b_N[2])

    state_N = y_N.*a_N.*b_N

    state_grid = gridmake(y_grid, a_grid, b_grid)
    state_gridT = gridmake(y_gridT, a_gridT, b_gridT)

    new(y_min, y_max, y_N, y_grid, y_gridT,
      a_min, a_max, a_N, a_grid, a_gridT,
      b_min, b_max, b_N, b_grid, b_gridT,
      state_N, state_grid, state_gridT)

  end

end

#= Misc Utilities =#

function cheby_nodes!(lower::Float64,upper::Float64,node_size::Int64)

  nodes_unit = zeros(node_size)
  nodes_adjust = zeros(node_size)

  # compute nodes on [-1,1]
  for k in 1:node_size
    nodes_unit[k] = -cos(((2*k-1)/(2*node_size))*pi)
  end

  # adjust nodes to [lower,upper]
  for k in 1:node_size
    nodes_adjust[k] = (nodes_unit[k]+1)*((upper-lower)/2) + lower
  end

  return nodes_adjust

end

#= Exogenous Processes =#

# human capital production function (period specific)

function HC_prod(b::Float64, x::Float64, shock::Float64,
  iota0::Float64, iota1::Float64, iota2::Float64, iota3::Float64)

  bprime = exp(iota0 + iota1*log(b) + iota2*log(x) + iota3*log(b)*log(x))*exp(shock)

  return bprime

end

# income evolution

function Y_evol(y::Float64, rho_y::Float64, shock::Float64)

  lnyprime = rho_y*log(y) + shock

  yprime = exp(lnyprime)

  # adjust for yearly (from six-year value)
  yprime_yearly = yprime/6.

  return yprime_yearly

end

#= Household Preferences and Tuition Optimization =#

## Reduced Form Tuition

# terminal utility

function u_T(y::Float64, a::Float64, b::Float64, alphaT1::Float64, alphaT2::Float64,
    beta0::Float64, beta1::Float64, beta2::Float64, tuition::Float64, r::Float64)

    # present value of income
    y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    # present value of tuition
    t_pv = dot(tuition*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    utility = alphaT1*log(y_pv + a - t_pv) + alphaT2*(beta0 + beta1*log(tuition) + beta2*log(tuition)*log(b))

end

# RF college quality optimal choice

function t_opt(y::Float64, a::Float64, b::Float64,
  alphaT1::Float64, alphaT2::Float64, beta0::Float64, beta1::Float64, beta2::Float64, r::Float64)

  # annualize r
  r_annual = r/6.

  pv_scale = dot(ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])
  y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

  t_star = alphaT2*(beta1+beta2*log(b))*(y_pv + a)/
        (pv_scale*(alphaT1 + alphaT2*(beta1+beta2*log(b))))

  return t_star

end

# terminal value closed form

function EV_T(y::Float64, a::Float64, b::Float64, paramsdec::ParametersDec)

  t_star = max(t_opt(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2,
    paramsdec.beta0, paramsdec.beta1, paramsdec.beta2, paramsdec.r), 1.0)

  Value = paramsdec.B*u_T(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2,
    paramsdec.beta0, paramsdec.beta1, paramsdec.beta2, t_star, paramsdec.r)

  return Value

end

#= Bellman Operators =#

## First Period

function bellman_optim_child!(y::Float64, a::Float64, b::Float64,
  paramsdec::ParametersDec, paramsshock::ParametersShock;
  aprime_start=1., x_start=1., opt_code="neldermead", error_log_flag=0)

  # create endogenous maximum for next-period asset holdings
  aprime_max = y + (1+paramsdec.r)*a

  # define objective function
  function utility(choices)

   c = y - choices[1] - choices[2]

   if c > 0.0 && choices[1] >= 1.0 && choices[1] <= aprime_max && choices[2] >= 1.

     # calculate possible next-period (y,b) values
     y_b_array = zeros(paramsshock.eps_joint_discrete_N,2)

     for j in 1:paramsshock.eps_joint_discrete_N
       y_b_array[j,1] = Y_evol(y, paramsdec.rho_y, paramsshock.eps_joint_discrete_range[j,1])
       y_b_array[j,2] = HC_prod(b, choices[2], paramsshock.eps_joint_discrete_range[j,2],
         paramsdec.iota0, paramsdec.iota1, paramsdec.iota2, paramsdec.iota3)
     end

     Vprime_array_exact = zeros(paramsshock.eps_joint_discrete_N)

     for j in 1:paramsshock.eps_joint_discrete_N
       Vprime_array_exact[j] = EV_T(y_b_array[j,1], choices[1], y_b_array[j,2], paramsdec)
     end

     value_out = log(c) + paramsdec.beta*dot(Vprime_array_exact,paramsshock.eps_joint_dist_discrete)

   else

     value_out = -c^2

   end

    return -1*value_out

  end

  # solve optimization problem
  if opt_code == "neldermead"
   opt_agent = optimize(utility, [aprime_start, x_start], show_trace=false, iterations=5000)
  elseif opt_code == "lbfgs"
   opt_agent = optimize(utility, [aprime_start, x_start], LBFGS())
  elseif opt_code == "simulatedannealing"
   opt_agent = optimize(utility, [aprime_start, x_start], method=SimulatedAnnealing(), iterations=5000)
  else
   throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
  end

  # error check, allow to continue and log error if flag is 1
  # if (opt_agent.g_converged == false) || (opt_agent.g_converged == true && opt_agent.iterations == 0)
  #  if error_log_flag == 0
  #    error("did not converge: ",string(y," ",a," ",b))
  #  else
  #    error_return_state = [y a b]
  #  end
  # else
  #  error_return_state = Any[]
  # end

  return -1*opt_agent.minimum, opt_agent.minimizer, opt_agent.g_converged, opt_agent.iterations, error_return_state

end
