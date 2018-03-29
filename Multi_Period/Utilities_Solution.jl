#= Utilities for Solution of Model =#

using Distributions
using GridInterpolations
using QuantEcon: gridmake

#= Parameters Structures =#

## Decision Problem Parameters

mutable struct ParametersDec
  beta :: Float64 ## discount rate
  r :: Float64 ## interest rate
  B :: Float64 ## temporary continuation value shifter

  alpha1 :: Float64 ## t-period utility parameter
  alpha2 :: Float64 ## t-period utility parameter
  alpha3 :: Float64 ## t-period utility parameter

  alphaT1 :: Float64 ## T-period utility parameter (RF choice)
  alphaT2 :: Float64 ## T-period utility parameter (RF choice)

  gammaT1 :: Float64 ## T-period utility parameter (DCP)
  gammaT2 :: Float64 ## T-period utility parameter (DCP)
  gammaT3 :: Float64 ## T-period utility parameter (DCP)
  gammaT4 :: Float64 ## T-period utility parameter (DCP)

  iota0 :: Array{Float64} ## Vector of HC production function parameters
  iota1 :: Array{Float64} ## Vector of HC production function parameters
  iota2 :: Array{Float64} ## Vector of HC production function parameters
  iota3 :: Array{Float64} ## Vector of HC production function parameters

  beta0 :: Float64 ## RF college quality parameter
  beta1 :: Float64 ## RF college quality parameter
  beta2 :: Float64 ## RF college quality parameter

  rho_y :: Float64 ## income persistence parameter

  function ParametersDec(;beta=0.902, r=0.06, B=1.,
    alpha1=0.3, alpha2=0.3, alpha3=1-alpha1-alpha2, alphaT1=0.5, alphaT2=1-alphaT1,
    gammaT1=0.25, gammaT2=0.25, gammaT3=0.25, gammaT4=1-gammaT1-gammaT2-gammaT3,
    iota0=[1.87428633 1.88977362 0.84394021], iota1=[0.42122805 0.43082281 0.64656581],
    iota2=[0.05979631 0.05642741 0.09121635], iota3=[0.0 0.0 0.0],
    beta0=4.35, beta1=-0.003, beta2=0.0037, rho_y=0.997)

    new(beta, r, B, alpha1, alpha2, alpha3, alphaT1, alphaT2,
    gammaT1, gammaT2, gammaT3, gammaT4,
    iota0, iota1, iota2, iota3, beta0, beta1, beta2, rho_y)

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
  y_grid1 :: Array ## income grid in t=1
  y_grid2 :: Array ## income grid in t=2
  y_grid3 :: Array ## income grid in t=3
  y_grid4 :: Array ## income grid in t=4

  a_min :: Array{Float64} ## asset grid lower bound
  a_max :: Array{Float64} ## asset grid upper bound
  a_N :: Array{Int64} ## number of asset grid points
  a_grid1 :: Array ## asset grid in t=1
  a_grid2 :: Array ## asset grid in t=2
  a_grid3 :: Array ## asset grid in t=3
  a_grid4 :: Array ## asset grid in t=4

  b_min :: Array{Float64} ## HC grid lower bound
  b_max :: Array{Float64} ## HC grid upper bound
  b_N :: Array{Int64} ## number of HC grid points
  b_grid1 :: Array ## HC grid in t=1
  b_grid2 :: Array ## HC grid in t=2
  b_grid3 :: Array ## HC grid in t=3
  b_grid4 :: Array ## HC grid in t=4

  state_N :: Array{Int64} ## number of state grid points
  state_grid1 :: Array ## state grid in t=1
  state_grid2 :: Array ## state grid in t=2
  state_grid3 :: Array ## state grid in t=3
  state_grid4 :: Array ## state grid in t=4
  state_grid_interp2 :: GridInterpolations.RectangleGrid ## state grid used for interpolations V_2
  state_grid_interp3 :: GridInterpolations.RectangleGrid ## state grid used for interpolations V_3

  function StateGrids(;y_min=[1000.0 1000.0 1000.0 1000.0], y_max=[150000.0 150000.0 150000.0 150000.0], y_N=[5 5 5 5],
              a_min=[1.0 1.0 1.0 1.0], a_max=[100000.0 100000.0 100000.0 100000.0], a_N=[5 5 5 5],
              b_min=[5.0 15.0 20.0 20.0], b_max=[60.0 70.0 80.0 100.0], b_N=[5 5 5 5])

    # create grids (uniform)

    # y_grid = linspace(y_min, y_max, y_N)
    # a_grid = linspace(a_min, a_max, a_N)
    # b_grid = linspace(b_min, b_max, b_N)

    # create grids (chebyshev spacing)

    y_grid1 = cheby_nodes!(y_min[1], y_max[1], y_N[1])
    y_grid2 = cheby_nodes!(y_min[2], y_max[2], y_N[2])
    y_grid3 = cheby_nodes!(y_min[3], y_max[3], y_N[3])
    y_grid4 = cheby_nodes!(y_min[4], y_max[4], y_N[4])

    a_grid1 = cheby_nodes!(a_min[1], a_max[1], a_N[1])
    a_grid2 = cheby_nodes!(a_min[2], a_max[2], a_N[2])
    a_grid3 = cheby_nodes!(a_min[3], a_max[3], a_N[3])
    a_grid4 = cheby_nodes!(a_min[4], a_max[4], a_N[4])

    b_grid1 = cheby_nodes!(b_min[1], b_max[1], b_N[1])
    b_grid2 = cheby_nodes!(b_min[2], b_max[2], b_N[2])
    b_grid3 = cheby_nodes!(b_min[3], b_max[3], b_N[3])
    b_grid4 = cheby_nodes!(b_min[4], b_max[4], b_N[4])

    state_N = y_N.*a_N.*b_N

    state_grid1 = gridmake(y_grid1, a_grid1, b_grid1)
    state_grid2 = gridmake(y_grid2, a_grid2, b_grid2)
    state_grid3 = gridmake(y_grid3, a_grid3, b_grid3)
    state_grid4 = gridmake(y_grid4, a_grid4, b_grid4)

    state_grid_interp2 = RectangleGrid(y_grid2, a_grid2, b_grid2)
    state_grid_interp3 = RectangleGrid(y_grid3, a_grid3, b_grid3)

    new(y_min, y_max, y_N, y_grid1, y_grid2, y_grid3, y_grid4,
      a_min, a_max, a_N, a_grid1, a_grid2, a_grid3, a_grid4,
      b_min, b_max, b_N, b_grid1, b_grid2, b_grid3, b_grid4,
      state_N, state_grid1, state_grid2, state_grid3, state_grid4,
      state_grid_interp2, state_grid_interp3)

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

  return yprime

end

#= Household Preferences and Tuition Optimization =#

# non-terminal period utility

function u_t(c::Float64, b::Float64, alpha1::Float64, alpha2::Float64, alpha3::Float64; l=8)

  utility = alpha1*log(c) + alpha2*log(b) + alpha3*log(l)

  return utility

end

# terminal utility

function u_T(y::Float64, a::Float64, b::Float64, alphaT1::Float64, alphaT2::Float64,
    beta0::Float64, beta1::Float64, beta2::Float64, tuition::Float64, r::Float64)

    # present value of income
    y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    # present value of tuition
    t_pv = dot(tuition*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    utility = alphaT1*log(y_pv + a - t_pv) + alphaT2*(beta0 + beta1*log(t) + beta2*log(tuition)*log(b))

end

## Reduced Form Tuition

# RF college quality optimal choice

function t_opt(y::Float64, a::Float64, b::Float64,
  alphaT1::Float64, alphaT2::Float64, beta0::Float64, beta1::Float64, beta2::Float64, r::Float64)

  pv_scale = dot(ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])
  y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

  t_star = alphaT2*(beta1+beta2*log(b))*(y_pv + a)/
        (pv_scale*(alphaT1 + alphaT2*(beta1+beta2*log(b))))

  return t_star

end

#= Terminal Value Closed Form =#

function EV_T(y::Float64, a::Float64, b::Float64, paramsdec::ParametersDec)

  t_star = max(t_opt(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2,
    paramsdec.beta0, paramsdec.beta1, paramsdec.beta2, paramsdec.r), 1.0)

  Value = paramsdec.B*u_T(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2,
    paramsdec.beta0, paramsdec.beta1, paramsdec.beta2, t_star, paramsdec.r)

  return Value

end

## Discrete Choice Problem

# choice-specific utility

function u_j(y::Float64, a::Float64, b::Float64, alphaT1::Float64, alphaT2::Float64,
    alphaT3::Float64, alphaT4::Float64, tuition::Float64, quality::Float64, r::Float64)

    # present value of income
    y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    # present value of tuition
    t_pv = dot(tuition*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    utility = alphaT1*log(y_pv + a - t_pv) + alphaT2*log(quality) +
      alphaT3*log(b) + alphaT4*log(quality)*log(b)

    return utility

end

# ex ante value

function Emax_T(y::Float64, a::Float64, b::Float64, t0::Float64, tpub::Float64, tpri::Float64,
  q0::Float64, qpub::Float64, qpri::Float64, paramsdec::ParametersDec)

  u0 = u_j(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2, paramsdec.alphaT3, paramsdec.alphaT4,
    t0, q0, r)
  upub = u_j(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2, paramsdec.alphaT3, paramsdec.alphaT4,
    tpub, qpub, r)
  upri = u_j(y, a, b, paramsdec.alphaT1, paramsdec.alphaT2, paramsdec.alphaT3, paramsdec.alphaT4,
    tpri, qpri, r)

  value = log(exp(u0)+exp(upub)+exp(upri))

  return value

end

#= Bellman Operators =#

## TEMP NOTE: assuming terminal marginal value of human capital is alphaT2. This is
##            inconsistent with model as written due to complementarity. I will change this
##            but for the moment roll with it. But understand that the problem is that
##            explicit formulation for x* is incorrect.

## T-1 Period

function bellman_Tminus1_gridsearch!(y::Float64, a::Float64, b::Float64,
  paramsdec::ParametersDec, paramsshock::ParametersShock;
  aprime_min=1., aprime_N=10)

  # assume monotone policy rule, stop search when value falls
  decreasing_flag = 0.

  # create endogenous maximum for next-period asset holdings
  aprime_max = y + (1+paramsdec.r)*a

  # create control grid
  aprime_grid = linspace(aprime_min, aprime_max, aprime_N)

  value = -Inf
  aprime_star = 0.
  x_star = 0.

  # search over aprime, using explicit formulation for x conditional on aprime
  for i in 1:aprime_N

    if decreasing_flag == 0.

      aprime_choice = aprime_grid[i]
      x_choice = (y + (1+paramsdec.r)*a - aprime_choice)*
        (paramsdec.beta*paramsdec.alphaT2*paramsdec.iota2[3])/
        (paramsdec.alpha1 + paramsdec.beta*paramsdec.alphaT2*paramsdec.iota2[3])

      c = y + (1+paramsdec.r)*a - x_choice - aprime_choice

      if c > 0.0 && x_choice >= 1.0 && aprime_choice >= 1.0

        # calculate possible next-period (y,b) values given shocks
        y_b_array = zeros(paramsshock.eps_joint_discrete_N,2)

        for j in 1:paramsshock.eps_joint_discrete_N
          y_b_array[j,1] = Y_evol(y, paramsdec.rho_y, paramsshock.eps_joint_discrete_range[j,1])
          y_b_array[j,2] = HC_prod(b, x_choice, paramsshock.eps_joint_discrete_range[j,2],
            paramsdec.iota0[3], paramsdec.iota1[3], paramsdec.iota2[3], paramsdec.iota3[3])
        end

        Vprime_array_exact = zeros(paramsshock.eps_joint_discrete_N)

        for j in 1:paramsshock.eps_joint_discrete_N
          Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime_choice, y_b_array[j,2], paramsdec)
        end

        value_choice = u_t(c, b, paramsdec.alpha1, paramsdec.alpha2, paramsdec.alpha3) +
          paramsdec.beta*dot(Vprime_array_exact,paramsshock.eps_joint_dist_discrete)

        if value_choice > value
          value = value_choice
          aprime_star = aprime_choice
          x_star = x_choice
        elseif value_choice < value
          decreasing_flag = 1.
        end


      end
    end

  end

  return value, aprime_star, x_star

end

## t < T-1 periods

function bellman_t_gridsearch!(y::Float64, a::Float64, b::Float64, t::Int64, Vprime::Array,
    Vprime_domain_interp::GridInterpolations.RectangleGrid,
    paramsdec::ParametersDec, paramsshock::ParametersShock;
    aprime_min=1., aprime_N=10)

    # assume monotone policy rule, stop search when value falls
    decreasing_flag = 0.

    # create endogenous maximum for next-period asset holdings
    aprime_max = y + (1+paramsdec.r)*a

    # create control grid
    aprime_grid = linspace(aprime_min, aprime_max, aprime_N)

    value = -Inf
    aprime_star = 0.
    x_star = 0.

    # search over aprime, using explicit formulation for x conditional on aprime
    for i in 1:aprime_N

      if decreasing_flag == 0.

        aprime_choice = aprime_grid[i]
        x_choice = (y + (1+paramsdec.r)*a - aprime_choice)*
          (paramsdec.beta*(paramsdec.alpha2 + paramsdec.beta*paramsdec.alphaT2)*paramsdec.iota2[t])/
          (paramsdec.alpha1 + paramsdec.beta*(paramsdec.alpha2 + paramsdec.beta*paramsdec.alphaT2)*paramsdec.iota2[t])

        c = y + (1+paramsdec.r)*a - x_choice - aprime_choice

        if c > 0.0 && x_choice >= 1.0 && aprime_choice >= 1.0

          # calculate possible next-period (y,b) values given shocks
          y_b_array = zeros(paramsshock.eps_joint_discrete_N,2)

          for j in 1:paramsshock.eps_joint_discrete_N
            y_b_array[j,1] = Y_evol(y, paramsdec.rho_y, paramsshock.eps_joint_discrete_range[j,1])
            y_b_array[j,2] = HC_prod(b, x_choice, paramsshock.eps_joint_discrete_range[j,2],
              paramsdec.iota0[3], paramsdec.iota1[3], paramsdec.iota2[3], paramsdec.iota3[3])
          end

          Vprime_array_exact = zeros(paramsshock.eps_joint_discrete_N)

          for j in 1:paramsshock.eps_joint_discrete_N
            Vprime_array_exact[j] = interpolate(Vprime_domain_interp, Vprime, [y_b_array[j,1], aprime_choice, y_b_array[j,2]])
          end

          value_choice = u_t(c, b, paramsdec.alpha1, paramsdec.alpha2, paramsdec.alpha3) +
            paramsdec.beta*dot(Vprime_array_exact,paramsshock.eps_joint_dist_discrete)

          if value_choice > value
            value = value_choice
            aprime_star = aprime_choice
            x_star = x_choice
          elseif value_choice < value
            decreasing_flag = 1.
          end


        end
      end

    end

    return value, aprime_star, x_star

end

#= Backward Induction =#

## For Specific HH Initial Condition and Preference Parameters

function back_induction_hh!(y0::Float64, a0::Float64, b0::Float64,
  paramsdec::ParametersDec, paramsshock::ParametersShock, stategrids::StateGrids;
  t0::Int64=1, aprime_min::Float64=1., aprime_N::Int64=10)

  # check that initial period is t=1, 2, or 3
  if (t0 < 1) || (t0 > 3)
    throw(error("Initial period must be t0=1, 2, or 3"))
  end

  # set terminal period T
  T = 4

  # initialize value and choices for each grid point in each period from second-period on
  V_array = Array{Array{Float64}}(T-t0+1)
  aprime_array = Array{Array{Float64}}(T-t0)
  x_array = Array{Array{Float64}}(T-t0)

  # println(4)

  # solve last period on last period grid
  V_T = fill(-Inf, stategrids.state_N[4])
  for i in 1:stategrids.state_N[4]
    # println(i)
    V_T[i] = EV_T(stategrids.state_grid4[i,1], stategrids.state_grid4[i,2], stategrids.state_grid4[i,3], paramsdec)
  end

  # store terminal value
  V_array[T-t0+1] = V_T

  # solve backwards until period t0 + 1
  if t0 < 3

    for t_index in 1:T-1-t0

      # keep track of period time
      t_calendar = T-t_index

      # println(t_calendar)

      V_t = fill(-Inf, stategrids.state_N[t_calendar])
      aprime_t = zeros(stategrids.state_N[t_calendar])
      x_t = zeros(stategrids.state_N[t_calendar])

      for i in 1:stategrids.state_N[t_calendar]
        # println(i)

        if t_calendar == 3
          opt_sol = bellman_Tminus1_gridsearch!(stategrids.state_grid3[i,1], stategrids.state_grid3[i,2], stategrids.state_grid3[i,3],
            paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
        elseif t_calendar == 2
          opt_sol = bellman_t_gridsearch!(stategrids.state_grid2[i,1], stategrids.state_grid2[i,2], stategrids.state_grid2[i,3],
            2, V_array[t_calendar+1], stategrids.state_grid_interp3, paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
        else
          throw(error("t=2 or 3"))
        end

        # store value and choices_x
        V_t[i] = opt_sol[1]
        aprime_t[i] = opt_sol[2]
        x_t[i] = opt_sol[3]

      end

      V_array[t_calendar-t0+1] = V_t
      aprime_array[t_calendar-t0+1] = aprime_t
      x_array[t_calendar-t0+1] = x_t

    end

  end

  # compute initial choices
  if t0 == 1
    opt_sol = bellman_t_gridsearch!(y0, a0, b0, 1, V_array[2], stategrids.state_grid_interp2,
      paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
  elseif t0 == 2
    opt_sol = bellman_t_gridsearch!(y0, a0, b0, 2, V_array[3], stategrids.state_grid_interp3,
      paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
  elseif t0 == 3
    opt_sol = bellman_Tminus1_gridsearch!(y0, a0, b0,
      paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
  end

  V0 = opt_sol[1]
  aprime0 = opt_sol[2]
  x0 = opt_sol[3]

  V_array[1] = [V0]
  aprime_array[1] = [aprime0]
  x_array[1] = [x0]

  return V0, aprime0, x0, V_array, aprime_array, x_array

end

## For All Initial States and Given Preference Parameters

# function back_induction!(paramsdec::ParametersDec, paramsshock::ParametersShock, stategrids::StateGrids;
#   aprime_min::Float64=1., aprime_N::Int64=10)
#
#   # set terminal period T
#   T = 4
#
#   # initialize value and choices for each grid point in each period from second-period on
#   V_array = Array{Array{Float64}}(T)
#   aprime_array = Array{Array{Float64}}(T-1)
#   x_array = Array{Array{Float64}}(T-1)
#
#   println(4)
#
#   # solve last period on last period grid
#   V_T = fill(-Inf, stategrids.state_N[4])
#   for i in 1:stategrids.state_N[4]
#     # println(i)
#     V_T[i] = EV_T(stategrids.state_grid4[i,1], stategrids.state_grid4[i,2], stategrids.state_grid4[i,3], paramsdec)
#   end
#
#   # store terminal value
#   V_array[T] = V_T
#
#   # solve backwards until period t0 + 1
#   for t_index in 1:T-1
#
#     # keep track of period time
#     t_calendar = T-t_index
#
#     println(t_calendar)
#
#     V_t = fill(-Inf, stategrids.state_N[t_calendar])
#     aprime_t = zeros(stategrids.state_N[t_calendar])
#     x_t = zeros(stategrids.state_N[t_calendar])
#
#     for i in 1:stategrids.state_N[t_calendar]
#       # println(i)
#
#       if t_calendar == 3
#         opt_sol = bellman_Tminus1_gridsearch!(stategrids.state_grid3[i,1], stategrids.state_grid3[i,2], stategrids.state_grid3[i,3],
#           paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
#       elseif t_calendar == 2
#         opt_sol = bellman_t_gridsearch!(stategrids.state_grid2[i,1], stategrids.state_grid2[i,2], stategrids.state_grid2[i,3],
#           2, V_array[t_calendar+1], stategrids.state_grid_interp3, paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
#       elseif t_calendar == 1
#         opt_sol = bellman_t_gridsearch!(stategrids.state_grid1[i,1], stategrids.state_grid1[i,2], stategrids.state_grid1[i,3],
#           1, V_array[t_calendar+1], stategrids.state_grid_interp2, paramsdec, paramsshock, aprime_min=aprime_min, aprime_N=aprime_N)
#       else
#         throw(error("t=1, 2 or 3"))
#       end
#
#       # store value and choices_x
#       V_t[i] = opt_sol[1]
#       aprime_t[i] = opt_sol[2]
#       x_t[i] = opt_sol[3]
#
#     end
#
#     V_array[t_calendar] = V_t
#     aprime_array[t_calendar] = aprime_t
#     x_array[t_calendar] = x_t
#
#   end
#
#   return V_array, aprime_array, x_array
#
# end
