#= Interpolation Testing =#

using Optim
using Distributions
using GridInterpolations
using QuantEcon: gridmake
using PyPlot
using DataFrames
using KernelDensity
using LatexPrint

#= Parameters Structure =#

mutable struct Parameterst
  beta :: Float64 ## discount rate
  r :: Float64 ## interest rate
  B :: Float64 ## temporary continuation value shifter

  iota0 :: Array{Float64} ## Vector of HC production function parameters
  iota1 :: Array{Float64} ## Vector of HC production function parameters
  iota2 :: Array{Float64} ## Vector of HC production function parameters
  iota3 :: Array{Float64} ## Vector of HC production function parameters

  rho_y :: Float64 ## Income persistence parameter

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

  alpha01 :: Float64 # t-period utility parameter
  alpha02 :: Float64 # t-period utility parameter

  alpha1 :: Float64 ## T-period utility parameter
  alpha2 :: Float64 ## T-period utility parameter

  y_min :: Float64 ## income grid lower bound
  y_max :: Float64 ## income grid upper bound
  y_N :: Int64 ## number of income grid points
  y_grid :: Array ## income grid

  a_min :: Float64 ## asset grid lower bound
  a_max :: Float64 ## asset grid upper bound
  a_N :: Int64 ## number of asset grid points
  a_grid :: Array ## asset grid

  b_min :: Float64 ## HC grid lower bound
  b_max :: Float64 ## HC grid upper bound
  b_N :: Int64 ## number of HC grid points
  b_grid :: Array ## HC grid

  state_N :: Int64 ## number of state grid points
  state_grid :: Array ## state grid
  state_grid_interp :: GridInterpolations.RectangleGrid ## state grid used for interpolations

  beta0 :: Float64 ## RF college quality parameter
  beta1 :: Float64 ## RF college quality parameter
  beta2 :: Float64 ## RF college quality parameter

  function Parameterst(;beta=0.902, r=0.06, B=1.,
    iota0=[1.87428633 1.88977362 0.84394021], iota1=[0.42122805 0.43082281 0.64656581], iota2=[0.05979631 0.05642741 0.09121635], iota3=[0.0 0.0 0.0],
              rho_y=1.007768, eps_b_mu=0.0, eps_b_var=0.022, eps_b_discrete_N=10,
              eps_y_mu=0.0, eps_y_var=0.2513478, eps_y_discrete_N=10, sd_num_b=2.0, sd_num_y=2.0,
              alpha01=0.5, alpha02=0.5, alpha1=0.32, alpha2=1-0.32,
              y_min=10000.0, y_max=100000.0, y_N=5,
              a_min=1000.0, a_max=100000.0, a_N=5,
              b_min=1.0, b_max=84.0, b_N=5,
              beta0=4.35, beta1=-0.003, beta2=0.0037)

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

    # create grids (uniform)

    # y_grid = linspace(y_min, y_max, y_N)
    # a_grid = linspace(a_min, a_max, a_N)
    # b_grid = linspace(b_min, b_max, b_N)

    # create grids (chebyshev spacing)

    y_grid = cheby_nodes!(y_min, y_max, y_N)
    a_grid = cheby_nodes!(a_min, a_max, a_N)
    b_grid = cheby_nodes!(b_min, b_max, b_N)

    state_N = y_N*a_N*b_N

    state_grid = gridmake(y_grid, a_grid, b_grid)
    state_grid_interp = RectangleGrid(y_grid, a_grid, b_grid)


    new(beta, r, B, iota0, iota1, iota2, iota3, rho_y,
      eps_b_dist, eps_b_mu, eps_b_var, eps_b_discrete_min, eps_b_discrete_max, eps_b_discrete_N,
      eps_b_discrete_range, eps_b_dist_discrete,
      eps_y_dist, eps_y_mu, eps_y_var, eps_y_discrete_min, eps_y_discrete_max, eps_y_discrete_N,
      eps_y_discrete_range, eps_y_dist_discrete,
      eps_joint_discrete_N, eps_joint_discrete_range, eps_joint_dist_discrete,
      alpha01, alpha02, alpha1, alpha2,
      y_min, y_max, y_N, y_grid,
      a_min, a_max, a_N, a_grid,
      b_min, b_max, b_N, b_grid,
      state_N, state_grid, state_grid_interp,
      beta0, beta1, beta2)

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

function HC_prod(b::Float64, x::Float64, shock::Float64, t::Int64, paramst::Parameterst)

  bprime = exp(paramst.iota0[t] + paramst.iota1[t]*log(b) + paramst.iota2[t]*log(x) +
                  paramst.iota3[t]*log(b)*log(x))*exp(shock)

  return bprime

end

# income evolution

function Y_evol(y::Float64, shock::Float64, paramst::Parameterst)

  lnyprime = paramst.rho_y*log(y) + shock

  yprime = exp(lnyprime)

  return yprime

end

#= Household Optimization =#

# non-terminal period utility

function u_t(c::Float64, b::Float64, alpha01::Float64, alpha02::Float64)

  utility = alpha01*log(c) + alpha02*log(b)

  return utility

end

# terminal utility

function u_j(y::Float64, a::Float64, b::Float64, alpha1::Float64, alpha2::Float64,
    beta0::Float64, beta1::Float64, beta2::Float64, t::Float64, r::Float64)

    # present value of income
    y_pv = dot(y*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    # present value of tuition
    t_pv = dot(t*ones(4),(1/(1+r)).^[1.0 2.0 3.0 4.0])

    utility = alpha1*log(y_pv + a - t_pv) + alpha2*(beta0 + beta1*log(t) + beta2*log(t)*log(b))

end

# RF college quality

function t_opt(y::Float64, a::Float64, b::Float64, paramst::Parameterst)

  pv_scale = dot(ones(4),(1/(1+paramst.r)).^[1.0 2.0 3.0 4.0])
  y_pv = dot(y*ones(4),(1/(1+paramst.r)).^[1.0 2.0 3.0 4.0])

  t_star = paramst.alpha2*(paramst.beta1+paramst.beta2*log(b))*(y_pv + a)/
        (pv_scale*(paramst.alpha1 + paramst.alpha2*(paramst.beta1+paramst.beta2*log(b))))

  return t_star

end

#= Terminal Value Closed Form =#

function EV_T(y::Float64, a::Float64, b::Float64, paramst::Parameterst)

  t_star = max(t_opt(y,a,b,paramst), 1.0)

  Value = paramst.B*u_j(y, a, b, paramst.alpha1, paramst.alpha2, paramst.beta0, paramst.beta1, paramst.beta2, t_star, paramst.r)

  return Value

end

#= Test Interpolation by Plotting =#

function interp_plot_test(y_grid_test, a_grid_test, b_grid_test, paramst::Parameterst;
  y_N_fixed=50, a_N_fixed=50, b_N_fixed=50)

  y_N = length(y_grid_test)
  a_N = length(a_grid_test)
  b_N = length(b_grid_test)

  # evaluate function exactly on state grid
  V_T = zeros(paramst.state_N)
  for i in 1:paramst.state_N
    V_T[i] = EV_T(paramst.state_grid[i,1], paramst.state_grid[i,2], paramst.state_grid[i,3], paramst)
  end

  # create test grid
  grid_test = gridmake(y_grid_test, a_grid_test, b_grid_test)

  # evaluate function exactly on test grid
  V_exact = zeros(length(grid_test[:,1]))
  for i in 1:length(grid_test[:,1])
    V_exact[i] = EV_T(grid_test[i,1], grid_test[i,2], grid_test[i,3], paramst)
  end

  # interpolate function on test grid using V_T
  V_interp = zeros(length(grid_test[:,1]))
  for i in 1:length(grid_test[:,1])
    V_interp[i] = interpolate(paramst.state_grid_interp, V_T, [grid_test[i,1], grid_test[i,2], grid_test[i,3]])
  end

  #  holding each state variable constant, plot value functions (exact and interpolated) in the other two directions

  V_exact_fix_y = V_exact[find(x->x==y_grid_test[y_N_fixed],grid_test[:,1])]
  V_interp_fix_y = V_interp[find(x->x==y_grid_test[y_N_fixed],grid_test[:,1])]

  V_exact_fix_a = V_exact[find(x->x==a_grid_test[a_N_fixed],grid_test[:,2])]
  V_interp_fix_a = V_interp[find(x->x==a_grid_test[a_N_fixed],grid_test[:,2])]

  V_exact_fix_b = V_exact[find(x->x==b_grid_test[b_N_fixed],grid_test[:,3])]
  V_interp_fix_b = V_interp[find(x->x==b_grid_test[b_N_fixed],grid_test[:,3])]

  plot_v_exact_fix_y = zeros(a_N, b_N)
  plot_v_interp_fix_y = zeros(a_N, b_N)

  plot_v_exact_fix_a = zeros(y_N, b_N)
  plot_v_interp_fix_a = zeros(y_N, b_N)

  plot_v_exact_fix_b = zeros(y_N, a_N)
  plot_v_interp_fix_b = zeros(y_N, a_N)

  for i in 1:length(plot_v_exact_fix_y)
    plot_v_exact_fix_y[i] = V_exact_fix_y[i]
    plot_v_interp_fix_y[i] = V_interp_fix_y[i]
  end

  for i in 1:length(plot_v_exact_fix_a)
    plot_v_exact_fix_a[i] = V_exact_fix_a[i]
    plot_v_interp_fix_a[i] = V_interp_fix_a[i]
  end

  for i in 1:length(plot_v_exact_fix_b)
    plot_v_exact_fix_b[i] = V_exact_fix_b[i]
    plot_v_interp_fix_b[i] = V_interp_fix_b[i]
  end

  y_fixed = figure()
  plot_surface(a_grid_test, b_grid_test, transpose(plot_v_exact_fix_y), rstride=2, edgecolors="k",
  cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  plot_surface(a_grid_test, b_grid_test, transpose(plot_v_interp_fix_y), rstride=2, edgecolors="k",
  cstride=2, alpha=0.8, linewidth=0.25)
  xlabel("a")
  ylabel("b")
  title(string("Value: y=",y_grid_test[y_N_fixed]))

  a_fixed = figure()
  plot_surface(y_grid_test, b_grid_test, transpose(plot_v_exact_fix_a), rstride=2, edgecolors="k",
  cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  plot_surface(y_grid_test, b_grid_test, transpose(plot_v_interp_fix_a), rstride=2, edgecolors="k",
  cstride=2, alpha=0.8, linewidth=0.25)
  xlabel("y")
  ylabel("b")
  title(string("Value: a=",a_grid_test[a_N_fixed]))

  b_fixed = figure()
  plot_surface(y_grid_test, a_grid_test, transpose(plot_v_exact_fix_b), rstride=2, edgecolors="k",
  cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  plot_surface(y_grid_test, a_grid_test, transpose(plot_v_interp_fix_b), rstride=2, edgecolors="k",
  cstride=2, alpha=0.8, linewidth=0.25)
  xlabel("y")
  ylabel("a")
  title(string("Value: b=",b_grid_test[b_N_fixed]))

  # gather max differences in each plotted surface

  max_exact_above_y = grid_test[find(x->x==maximum(V_exact_fix_y-V_interp_fix_y),V_exact_fix_y-V_interp_fix_y),:]
  max_exact_below_y = grid_test[find(x->x==minimum(V_exact_fix_y-V_interp_fix_y),V_exact_fix_y-V_interp_fix_y),:]

  max_exact_above_a = grid_test[find(x->x==maximum(V_exact_fix_a-V_interp_fix_a),V_exact_fix_a-V_interp_fix_a),:]
  max_exact_below_a = grid_test[find(x->x==minimum(V_exact_fix_a-V_interp_fix_a),V_exact_fix_a-V_interp_fix_a),:]

  max_exact_above_b = grid_test[find(x->x==maximum(V_exact_fix_b-V_interp_fix_b),V_exact_fix_b-V_interp_fix_b),:]
  max_exact_below_b = grid_test[find(x->x==minimum(V_exact_fix_b-V_interp_fix_b),V_exact_fix_b-V_interp_fix_b),:]

  max_above = [max_exact_above_y maximum(V_exact_fix_y-V_interp_fix_y);
              max_exact_above_a maximum(V_exact_fix_a-V_interp_fix_a);
              max_exact_above_b maximum(V_exact_fix_b-V_interp_fix_b)]

  max_below = [max_exact_below_y minimum(V_exact_fix_y-V_interp_fix_y);
              max_exact_below_a minimum(V_exact_fix_a-V_interp_fix_a);
              max_exact_below_b minimum(V_exact_fix_b-V_interp_fix_b)]

  # return V_exact, V_interp, exact_above, exact_below
  return V_exact, V_interp, max_above, max_below

end

#= Test Policy Differences Using Exact and Interpolated Functions =#

## 3rd Period Only, write both Optim and grid search functions

# Optim

function bellman_Optim_interp_3!(y::Float64, a::Float64, b::Float64, Vprime::Array, paramst::Parameterst;
  s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=0)

    # define objective function
    function utility(choices)

     c = y - choices[1] - choices[2]
     aprime = choices[1] + a*(1+paramst.r)

     if c > 0.0 && choices[2] >= 1.0 && aprime >= 1.0

       # calculate possible next-period (y,b) values
       y_b_array = zeros(paramst.eps_joint_discrete_N,2)

       for j in 1:paramst.eps_joint_discrete_N
         y_b_array[j,1] = Y_evol(y, paramst.eps_joint_discrete_range[j,1], paramst)
         y_b_array[j,2] = HC_prod(b, choices[2], paramst.eps_joint_discrete_range[j,2], 3, paramst)
       end

       Vprime_array_exact = zeros(paramst.eps_joint_discrete_N)
       Vprime_array_interp = zeros(paramst.eps_joint_discrete_N)

       for j in 1:paramst.eps_joint_discrete_N
         Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime, y_b_array[j,2], paramst)
         Vprime_array_interp[j] = interpolate(paramst.state_grid_interp, Vprime, [y_b_array[j,1], aprime, y_b_array[j,2]])
       end

       utility_exact = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_exact,paramst.eps_joint_dist_discrete)
       utility_interp = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_interp,paramst.eps_joint_dist_discrete)

       if interp_flag == 0
         utility_out = utility_exact
       elseif interp_flag == 1
         utility_out = utility_interp
       end

     else

       utility_out = -c^2

     end

      return -1*utility_out

   end

   # solve problem of agent
   if opt_code == "neldermead"
     opt_agent = optimize(utility, [s_start, x_start], show_trace=false, iterations=5000)
   elseif opt_code == "lbfgs"
     opt_agent = optimize(utility, [s_start, x_start], LBFGS())
   elseif opt_code == "simulatedannealing"
     opt_agent = optimize(utility, [s_start, x_start], method=SimulatedAnnealing(), iterations=5000)
   else
     throw(error("opt_code must be neldermead or lbfgs or simulatedannealing"))
   end

   # error check
   if (opt_agent.g_converged == false) || (opt_agent.g_converged == true && opt_agent.iterations == 0)
     error("did not converge: ",string(y," ",a," ",b))
   end

   return -1*opt_agent.minimum, opt_agent.minimizer, opt_agent.g_converged, opt_agent.iterations

end

# grid search

function bellman_grid_interp_3!(y::Float64, a::Float64, b::Float64, Vprime::Array, paramst::Parameterst;
  s_N=10, x_min=1.0, x_max=2500.0, x_N=10, interp_flag=0)

  s_min = -a*(1+paramst.r) + 1.
  s_max = y + a*(1+paramst.r) - 1.

  max_val = -Inf
  s_star = -Inf
  x_star = -Inf

  # create grids for grid search
  s_grid = linspace(s_min, s_max, s_N)
  x_grid = linspace(x_min, x_max, x_N)

  for s_index in 1:s_N

    s = s_grid[s_index]

    for x_index in 1:x_N

      x = x_grid[x_index]

      c = y - s - x
      aprime = s + a*(1+paramst.r)

      if c > 0.0 && aprime >= 1.0

        # calculate possible next-period (y,b) values
        y_b_array = zeros(paramst.eps_joint_discrete_N,2)

        for j in 1:paramst.eps_joint_discrete_N
          y_b_array[j,1] = Y_evol(y, paramst.eps_joint_discrete_range[j,1], paramst)
          y_b_array[j,2] = HC_prod(b, x, paramst.eps_joint_discrete_range[j,2], 3, paramst)
        end

        Vprime_array_exact = zeros(paramst.eps_joint_discrete_N)
        Vprime_array_interp = zeros(paramst.eps_joint_discrete_N)

        for j in 1:paramst.eps_joint_discrete_N
          Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime, y_b_array[j,2], paramst)
          Vprime_array_interp[j] = interpolate(paramst.state_grid_interp, Vprime, [y_b_array[j,1], aprime, y_b_array[j,2]])
        end

        utility_exact = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_exact,paramst.eps_joint_dist_discrete)
        utility_interp = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_interp,paramst.eps_joint_dist_discrete)

        if interp_flag == 0
          utility = utility_exact
        elseif interp_flag == 1
          utility = utility_interp
        end

        if utility > max_val
          max_val = utility
          s_star = s
          x_star = x
        end

      end

    end

  end

  return max_val, [s_star, x_star]

end

## Graph Value over Choices Holding One Choice Fixed, for Interpolated and Exact Vprime

# fix savings

function value_x(s_fix::Float64, y::Float64, a::Float64, b::Float64, Vprime::Array, paramst::Parameterst;
  x_min=1.0, x_max=2500.0, x_N=10)

  x_grid = linspace(x_min, x_max, x_N)

  utility_store_exact = zeros(x_N)
  utility_store_interp = zeros(x_N)

  for i in 1:x_N

    x = x_grid[i]

    c = y - s_fix - x
    aprime = s_fix + a*(1+paramst.r)

    if c > 0.0 && aprime >= 1.0

      # calculate possible next-period (y,b) values
      y_b_array = zeros(paramst.eps_joint_discrete_N,2)

      for j in 1:paramst.eps_joint_discrete_N
        y_b_array[j,1] = Y_evol(y, paramst.eps_joint_discrete_range[j,1], paramst)
        y_b_array[j,2] = HC_prod(b, x, paramst.eps_joint_discrete_range[j,2], 3, paramst)
      end

      Vprime_array_exact = zeros(paramst.eps_joint_discrete_N)
      Vprime_array_interp = zeros(paramst.eps_joint_discrete_N)

      for j in 1:paramst.eps_joint_discrete_N
        Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime, y_b_array[j,2], paramst)
        Vprime_array_interp[j] = interpolate(paramst.state_grid_interp, Vprime, [y_b_array[j,1], aprime, y_b_array[j,2]])
      end

      utility_store_exact[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_exact,paramst.eps_joint_dist_discrete)
      utility_store_interp[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_interp,paramst.eps_joint_dist_discrete)

    end

  end

  max_x_exact = x_grid[find(x->x==maximum(utility_store_exact),utility_store_exact)]
  max_x_interp = x_grid[find(x->x==maximum(utility_store_interp),utility_store_interp)]

  x_utility_fig = figure()
  plot(x_grid, utility_store_exact, label="value exact")
  plot(x_grid, utility_store_interp, label="value interp")
  ax = PyPlot.gca()
  ax[:legend](loc="lower right")
  title(string("Fixing s: ",s_fix))

  return max_x_exact[1], max_x_interp[1]

end

# fix investment

function value_s(x_fix::Float64, y::Float64, a::Float64, b::Float64, Vprime::Array, paramst::Parameterst;
  s_N=10)

  s_min = -a*(1+paramst.r) + 1.
  s_max = y + a*(1+paramst.r) - 1.

  s_grid = linspace(s_min, s_max, s_N)

  utility_store_exact = zeros(s_N)
  utility_store_interp = zeros(s_N)

  for i in 1:s_N

    s = s_grid[i]

    c = y - s - x_fix
    aprime = s + a*(1+paramst.r)

    if c > 0.0 && aprime >= 1.0

      # calculate possible next-period (y,b) values
      y_b_array = zeros(paramst.eps_joint_discrete_N,2)

      for j in 1:paramst.eps_joint_discrete_N
        y_b_array[j,1] = Y_evol(y, paramst.eps_joint_discrete_range[j,1], paramst)
        y_b_array[j,2] = HC_prod(b, x_fix, paramst.eps_joint_discrete_range[j,2], 3, paramst)
      end

      Vprime_array_exact = zeros(paramst.eps_joint_discrete_N)
      Vprime_array_interp = zeros(paramst.eps_joint_discrete_N)

      for j in 1:paramst.eps_joint_discrete_N
        Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime, y_b_array[j,2], paramst)
        Vprime_array_interp[j] = interpolate(paramst.state_grid_interp, Vprime, [y_b_array[j,1], aprime, y_b_array[j,2]])
      end

      utility_store_exact[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_exact,paramst.eps_joint_dist_discrete)
      utility_store_interp[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_interp,paramst.eps_joint_dist_discrete)

    end

  end

  s_utility_fig = figure()
  plot(s_grid, utility_store_exact, label="value exact")
  plot(s_grid, utility_store_interp, label="value interp")
  ax = PyPlot.gca()
  ax[:legend](loc="lower right")
  title(string("Fixing x: ",x_fix))

  max_s_exact = s_grid[find(x->x==maximum(utility_store_exact),utility_store_exact)]
  max_s_interp = s_grid[find(x->x==maximum(utility_store_interp),utility_store_interp)]

  return max_s_exact[1], max_s_interp[1]

end

# vary both s and x

function value_both(y::Float64, a::Float64, b::Float64, Vprime::Array, paramst::Parameterst;
  s_N=10, s_max=1000., x_min=1.0, x_max=2500.0, x_N=10, nosave_flag=0)

  if nosave_flag==0
    s_min = -a*(1+paramst.r) + 1.
    # s_max = y + a*(1+paramst.r) - 1.
    s_max = s_max
  elseif nosave_flag==1
    s_min = 0.
    s_max = s_max
  end

  s_grid = linspace(s_min, s_max, s_N)

  x_grid = linspace(x_min, x_max, x_N)

  s_x_grid = gridmake(s_grid, x_grid)

  utility_store_exact = zeros(x_N*s_N)
  utility_store_interp = zeros(x_N*s_N)

  for i in 1:x_N*s_N

    s = s_x_grid[i,1]
    x = s_x_grid[i,2]

    c = y - s - x
    aprime = s + a*(1+paramst.r)

    if c > 0.0 && aprime >= 1.0

      # calculate possible next-period (y,b) values
      y_b_array = zeros(paramst.eps_joint_discrete_N,2)

      for j in 1:paramst.eps_joint_discrete_N
        y_b_array[j,1] = Y_evol(y, paramst.eps_joint_discrete_range[j,1], paramst)
        y_b_array[j,2] = HC_prod(b, x, paramst.eps_joint_discrete_range[j,2], 3, paramst)
      end

      Vprime_array_exact = zeros(paramst.eps_joint_discrete_N)
      Vprime_array_interp = zeros(paramst.eps_joint_discrete_N)

      for j in 1:paramst.eps_joint_discrete_N
        Vprime_array_exact[j] = EV_T(y_b_array[j,1], aprime, y_b_array[j,2], paramst)
        Vprime_array_interp[j] = interpolate(paramst.state_grid_interp, Vprime, [y_b_array[j,1], aprime, y_b_array[j,2]])
      end

      utility_store_exact[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_exact,paramst.eps_joint_dist_discrete)
      utility_store_interp[i] = u_t(c,b,paramst.alpha01,paramst.alpha02) + paramst.beta*dot(Vprime_array_interp,paramst.eps_joint_dist_discrete)

    end

  end

  max_s_exact = s_x_grid[find(x->x==maximum(utility_store_exact),utility_store_exact),1]
  max_x_exact = s_x_grid[find(x->x==maximum(utility_store_exact),utility_store_exact),2]

  max_s_interp = s_x_grid[find(x->x==maximum(utility_store_interp),utility_store_interp),1]
  max_x_interp = s_x_grid[find(x->x==maximum(utility_store_interp),utility_store_interp),2]

  plot_v_exact = zeros(s_N, x_N)
  plot_v_interp = zeros(s_N, x_N)

  for i in 1:s_N*x_N
    plot_v_exact[i] = utility_store_exact[i]
    plot_v_interp[i] = utility_store_interp[i]
  end

  # calculate max difference in utility between exact and interp V_T

  max_interp_diff = maximum(abs(utility_store_exact-utility_store_interp))

  v_surface = figure()
  plot_surface(s_grid, x_grid, transpose(plot_v_exact), rstride=2, edgecolors="k",
  cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25, label="V_T exact")
  plot_surface(s_grid, x_grid, transpose(plot_v_interp), rstride=2, edgecolors="k",
  cstride=2, alpha=0.8, linewidth=0.25, label="V_T interp")
  xlabel("s")
  ylabel("x")
  title(string("Value: y=",y," a=",a," b=",b))

  return max_s_exact[1], max_x_exact[1], max_s_interp[1], max_x_interp[1], max_interp_diff

end

#= Initialize Parameters =#

param_vec = [0.902651, 0.463218, 8.58724, 0.999691, 0.253216, 0.0228707, 2.29149, 1.86123, 0.888602,
            0.516928, 0.452178, 0.56109, 0.052591, 0.0541078, 0.0975748, -0.000497188, 0.0101587, 0.0165175]

# initialize parameters structure

paramst = Parameterst(alpha01=param_vec[1], alpha02=1-param_vec[1],
              alpha1=param_vec[2], alpha2=1-param_vec[2], B=param_vec[3],
              rho_y=param_vec[4], eps_y_var=param_vec[5], eps_b_var=param_vec[6],
              iota0=param_vec[7:9], iota1=param_vec[10:12], iota2=param_vec[13:15],
              iota3=param_vec[16:18], y_N=5, a_N=5, b_N=5)

## Test Plotting Interpolations

y_grid = linspace(20000.,90000., 100)
a_grid = linspace(10000.,90000., 100)
b_grid = linspace(1.,120., 100)

V_exact_test, V_interp_test, max_above, max_below = interp_plot_test(y_grid, a_grid, b_grid, paramst,
    y_N_fixed=50, a_N_fixed=50, b_N_fixed=50)

## Testing Optimal Policies

# initialize parameters with finer grid

paramst_fine = Parameterst(alpha01=param_vec[1], alpha02=1-param_vec[1],
              alpha1=param_vec[2], alpha2=1-param_vec[2], B=param_vec[3],
              rho_y=param_vec[4], eps_y_var=param_vec[5], eps_b_var=param_vec[6],
              iota0=param_vec[7:9], iota1=param_vec[10:12], iota2=param_vec[13:15],
              iota3=param_vec[16:18], y_N=1000, a_N=1000, b_N=10, b_max=10.)

# calculate terminal period value exactly on state grid
Vprime = zeros(paramst_fine.state_N)
for i in 1:paramst_fine.state_N
  Vprime[i] = EV_T(paramst_fine.state_grid[i,1], paramst_fine.state_grid[i,2], paramst_fine.state_grid[i,3], paramst_fine)
end

# select agent to investigate

agent = 1

y = paramst.state_grid[agent,1]
a = paramst.state_grid[agent,2]
b = paramst.state_grid[agent,3]

# solve optim

optim_test = bellman_Optim_interp_3!(y, a, b, Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=0)

optim_test = bellman_Optim_interp_3!(y, a, b, Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=1)

# solve grid

grid_test = bellman_grid_interp_3!(y, a, b, Vprime, paramst_fine, s_N=500, x_N=300, x_min=1.0, x_max=4000.0, interp_flag=0)

# graph utility over choices, one at a time

value_x(-9550., y, a, b, Vprime, paramst_fine, x_min=1.0, x_max=2500.0, x_N=1000)

value_s(2323., y, a, b, Vprime, paramst_fine, s_N=1000)

# graph utility over both choices

value_both(y, a, b, Vprime, paramst_fine, s_N=100, s_max=10000., x_min=1.0, x_max=500.0, x_N=100, nosave_flag=0)

## Productivity Response Test

lo_prod = deepcopy(param_vec[15])
hi_prod = 0.2

# solve for all agents in COARSE state grid (use Optim only) under low and high values of last period investment productivity
# use fine grid for V_T interpolation, this is invariant to productiviy parameter

# solve problem, interpolating and using exact V_T separately, for low productivity

paramst_fine.iota2[3]=lo_prod

s_array_exact_lo = zeros(paramst.state_N)
x_array_exact_lo = zeros(paramst.state_N)

s_array_interp_lo = zeros(paramst.state_N)
x_array_interp_lo = zeros(paramst.state_N)

for j in 1:paramst.state_N
  s_array_exact_lo[j], x_array_exact_lo[j] = bellman_Optim_interp_3!(paramst.state_grid[j,1], paramst.state_grid[j,2], paramst.state_grid[j,3],
            Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=0)[2]
  s_array_interp_lo[j], x_array_interp_lo[j] = bellman_Optim_interp_3!(paramst.state_grid[j,1], paramst.state_grid[j,2], paramst.state_grid[j,3],
            Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=1)[2]
end

# solve problem, interpolating and using exact V_T separately, for high productivity

paramst_fine.iota2[3]=hi_prod

s_array_exact_hi = zeros(paramst.state_N)
x_array_exact_hi = zeros(paramst.state_N)

s_array_interp_hi = zeros(paramst.state_N)
x_array_interp_hi = zeros(paramst.state_N)

for j in 1:paramst.state_N
  s_array_exact_hi[j], x_array_exact_hi[j] = bellman_Optim_interp_3!(paramst.state_grid[j,1], paramst.state_grid[j,2], paramst.state_grid[j,3],
            Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=0)[2]
  s_array_interp_hi[j], x_array_interp_hi[j] = bellman_Optim_interp_3!(paramst.state_grid[j,1], paramst.state_grid[j,2], paramst.state_grid[j,3],
            Vprime, paramst_fine, s_start=1.0, x_start=1.0, opt_code="neldermead", interp_flag=1)[2]
end

# compare policy rules

delta_s_exact = s_array_exact_hi - s_array_exact_lo
delta_x_exact = x_array_exact_hi - x_array_exact_lo

delta_s_interp= s_array_interp_hi - s_array_interp_lo
delta_x_interp = x_array_interp_hi - x_array_interp_lo

# find agents who reduce investment when productivity increases

error_interp_index = find(x->x<=0.,delta_x_interp)

error_interp_i = paramst.state_grid[error_interp_index,:]

## Plot t=3 Value Over Choices, Exact and Interpolated, Under Both Productivity Levels

# select pathological agent

agent = 10

# low productivity

paramst_fine.iota2[3]=lo_prod
value_both(paramst.state_grid[agent,1], paramst.state_grid[agent,2], paramst.state_grid[agent,3], Vprime, paramst_fine,
  s_N=200, s_max=12000., x_min=1.0, x_max=1200.0, x_N=200, nosave_flag=1)

# high productivity

paramst_fine.iota2[3]=hi_prod
value_both(paramst.state_grid[agent,1], paramst.state_grid[agent,2], paramst.state_grid[agent,3], Vprime, paramst_fine,
  s_N=100, s_max=700., x_min=1.0, x_max=200.0, x_N=100, nosave_flag=1)
